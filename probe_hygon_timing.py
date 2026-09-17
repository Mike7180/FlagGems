# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which call in the training path waits for the device?

Temporary diagnostic, not part of the operator.

Three columns per row:

``cpu_idle``  time to enqueue the call, with an idle device.
``cpu_busy``  the same, but a 256 MiB memset is already running on the device.
              A call that only burns host time reads the same as ``cpu_idle``;
              a call that *waits for the device* reads the memset's duration
              instead.  This is the column that explains the benchmark: with a
              flush in flight, waiting shows up as device time, which is what
              ``do_bench`` ends up reporting.
``do_bench``  the harness' own measurement.

Run it on the benchmark host with ``python probe_hygon_timing.py`` and paste
the output back.
"""

import importlib
import os
import statistics
import time

import torch
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

try:
    hy = importlib.import_module("flag_gems.runtime.backend._hygon.ops.rrelu_with_noise")
except ImportError:  # a different backend is active; load the file from this tree
    import importlib.util

    _path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src",
        "flag_gems",
        "runtime",
        "backend",
        "_hygon",
        "ops",
        "rrelu_with_noise.py",
    )
    _spec = importlib.util.spec_from_file_location("hygon_rrelu", _path)
    hy = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(hy)

N = 4096
LOWER, UPPER = 0.125, 1.0 / 3.0

x = torch.randn(N, device=flag_gems.device)
noise = torch.zeros_like(x)
out = torch.empty_like(x)
tiny = torch.empty(2, dtype=torch.int64, device=flag_gems.device)
FLUSH = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=flag_gems.device)

gen = torch_device_fn.default_generators[torch_device_fn.current_device()]
st = gen.get_state()
sv = st.view(torch.int64)

BLOCK = hy._UNIFORM_HEURISTICS["BLOCK"]({"N": N})
NUM_WARPS = hy._UNIFORM_HEURISTICS["num_warps"]({"N": N})
GRID = (triton.cdiv(N, BLOCK * hy._TRAIN_UNROLL),)
INCR = triton.cdiv(N, hy._TRAIN_UNROLL)
_counter = [0]


def launch(seed, offset):
    with torch_device_fn.device(x.device):
        hy._rrelu_with_noise_train_contiguous_kernel[GRID](
            x,
            noise,
            out,
            N,
            float(LOWER),
            float(UPPER),
            seed,
            offset,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )


def from_counter():
    _counter[0] += 1
    launch((1 << 40) + _counter[0], 1 << 40)


def seeded_default():
    launch(*philox_backend_seed_offset(INCR))


def seeded_cached():
    launch(*philox_backend_seed_offset(INCR, generator=gen))


def philox_inline():
    """The same state arithmetic, without the module call or the generator lookup."""
    _counter[0] += 1
    base = 1 << 40
    return base + _counter[0], base + 2 * _counter[0]


def seeded_inline():
    launch(*philox_inline())


def py_work():
    """Pure host work, no torch at all: is the CPU itself slower when the
    device is busy?  If this row inflates, the extra time is not ours."""
    total = 0
    for i in range(12000):
        total += i * i
    return total


ROWS = [
    ("nothing", lambda: None),
    ("pure python work", py_work),
    ("launch fixed", lambda: launch(1234, 5678)),
    ("launch from py counter", from_counter),
    ("gen.get_state()", lambda: gen.get_state()),
    ("gen.set_state(st)", lambda: gen.set_state(st)),
    ("int(sv[0]), int(sv[1])", lambda: (int(sv[0]), int(sv[1]))),
    ("current_device()", lambda: torch_device_fn.current_device()),
    ("default_generators[dev]", lambda: torch_device_fn.default_generators[0]),
    ("philox (generator=None)", lambda: philox_backend_seed_offset(INCR)),
    ("philox (cached generator)", lambda: philox_backend_seed_offset(INCR, generator=gen)),
    ("launch seeded, generator=None", seeded_default),
    ("launch seeded, cached gen", seeded_cached),
    ("launch seeded, inlined", seeded_inline),
    ("empty_like", lambda: torch.empty_like(x)),
    ("patched op (fill_)", lambda: tiny.fill_(0)),
    ("hygon eval", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
    ("hygon train", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("hygon train inplace", lambda: hy.rrelu_with_noise_(x, noise, LOWER, UPPER, True)),
]


def flush_us(k=8):
    """How long the harness' own flush takes here.  Never patched: the
    benchmark passes ``exclude=["zero_"]``, so this is ATen's memset."""
    FLUSH.zero_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(k):
        torch.cuda.synchronize()
        start = time.perf_counter()
        FLUSH.zero_()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def cpu_idle(fn, n=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    torch.cuda.synchronize()
    return elapsed


def cpu_busy(fn, k=12):
    """CPU time of ``fn`` while a 256 MiB memset is running on the device."""
    FLUSH.zero_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(k):
        FLUSH.zero_()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(samples)


def bench(fn):
    try:
        done = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        return f"{done * 1e3:9.1f}"
    except Exception as exc:  # noqa: BLE001 - diagnostic
        return f"  FAILED({type(exc).__name__})"


print(f"device            {torch.cuda.get_device_name(0)}")
print(f"vendor            {flag_gems.vendor_name}")
print(f"torch/triton      {torch.__version__} / {triton.__version__}")
print(f"source            {hy.__file__}")
print(f"generator         {type(gen).__name__}, state on {st.device}")
print("cpu_busy is the column to read: it is the same call, with 256 MiB of\n"
      "memset already running.  A call that waits for the device reads large.")


def run(label):
    print(f"\n{label}   flush {flush_us():.1f} us")
    print("                            cpu_idle  cpu_busy  do_bench")
    for name, fn in ROWS:
        print(f"{name:24s} {cpu_idle(fn):8.1f}  {cpu_busy(fn):8.1f}  {bench(fn)}")


torch.cuda.synchronize()
run("outside use_gems()")
with flag_gems.use_gems(exclude=["zero_"]):
    run("inside use_gems()")
