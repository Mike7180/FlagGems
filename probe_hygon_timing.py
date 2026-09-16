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

"""Which piece of the training path carries the extra host time?

Temporary diagnostic, not part of the operator.

Every row below runs the same training kernel with the same grid and the same
arguments; only the work *around* the launch differs.  ``cpu`` is time to
enqueue with no synchronize in the loop.  ``do_bench`` is the harness' own
measurement -- it is the column the benchmark reports, and on this machine it
tracks ``cpu`` one for one above a floor.

Each row adds one piece to the row above, so the piece that costs what is
identifiable by subtraction.  ``launch from py counter`` is the ceiling: the
cheapest a correct launch could possibly be, with no generator and no tensor
work on the host at all.

Run it on the benchmark host with ``python probe_hygon_timing.py`` and paste
the output back.
"""

import importlib
import os
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

gen = torch_device_fn.default_generators[torch_device_fn.current_device()]
st = gen.get_state()
sv = st.view(torch.int64)
tiny = torch.empty(2, dtype=torch.int64, device=flag_gems.device)

BLOCK = hy._UNIFORM_HEURISTICS["BLOCK"]({"N": N})
NUM_WARPS = hy._UNIFORM_HEURISTICS["num_warps"]({"N": N})
GRID = (triton.cdiv(N, BLOCK * hy._TRAIN_UNROLL),)


def launch(seed, offset, device_cm=True):
    args = (x, noise, out, N, float(LOWER), float(UPPER), seed, offset)
    if device_cm:
        with torch_device_fn.device(x.device):
            hy._rrelu_with_noise_train_contiguous_kernel[GRID](
                *args, BLOCK=BLOCK, num_warps=NUM_WARPS
            )
    else:
        hy._rrelu_with_noise_train_contiguous_kernel[GRID](
            *args, BLOCK=BLOCK, num_warps=NUM_WARPS
        )


_counter = [0]


def from_counter():
    _counter[0] += 1
    launch((1 << 40) + _counter[0], 1 << 40)


def seeded():
    launch(*philox_backend_seed_offset(triton.cdiv(N, hy._TRAIN_UNROLL)))


def from_state_ints():
    launch(int(sv[0]), int(sv[1]))


_ROWS = [
    ("launch fixed", lambda: launch(1234, 5678)),
    ("launch fixed, no device CM", lambda: launch(1234, 5678, device_cm=False)),
    ("launch from py counter", from_counter),
    ("torch op then launch", lambda: (tiny.fill_(0), launch(1234, 5678))),
    ("empty_like then launch", lambda: (torch.empty_like(x), launch(1234, 5678))),
    ("get_state then launch", lambda: (gen.get_state(), launch(1234, 5678))),
    ("set_state then launch", lambda: (gen.set_state(st), launch(1234, 5678))),
    ("read ints then launch", from_state_ints),
    ("launch seeded (current)", seeded),
    ("hygon train", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("hygon train inplace", lambda: hy.rrelu_with_noise_(x, noise, LOWER, UPPER, True)),
]


def cpu_cost(fn, n=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    torch.cuda.synchronize()
    return elapsed


def bench(fn):
    try:
        done = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        return f"{done * 1e3:8.1f}"
    except Exception as exc:  # noqa: BLE001 - diagnostic
        return f"  FAILED({type(exc).__name__})"


print(f"device            {torch.cuda.get_device_name(0)}")
print(f"vendor            {flag_gems.vendor_name}")
print(f"torch/triton      {torch.__version__} / {triton.__version__}")
print(f"source            {hy.__file__}")
print(f"generator         {type(gen).__name__}, state on {st.device}")


def run():
    print("                            cpu(us)  do_bench(us)")
    for name, fn in _ROWS:
        print(f"{name:24s} {cpu_cost(fn):8.1f}  {bench(fn)}")


torch.cuda.synchronize()
print("\noutside use_gems()")
run()

with flag_gems.use_gems(exclude=["zero_"]):
    print("\ninside use_gems()")
    run()
