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

"""Where ``rrelu_with_noise`` spends its time on this machine.

Temporary diagnostic, not part of the operator.  Two sections:

1. ``do_bench`` of each piece, the same measurement the benchmark harness
   uses, so the numbers line up with its per-shape tables.
2. The same pieces timed while a large kernel is in flight.  A piece that
   blocks the CPU waiting for the device shows up as ~the duration of that
   kernel; one that does not costs its own time only.  This separates "slow
   Python" from "synchronizes", which ``do_bench`` alone cannot tell apart.

Run it on the benchmark host with ``python probe_hygon_timing.py`` and paste
the output back.
"""

import importlib
import statistics
import time

import torch
import triton

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import philox_backend_seed_offset

try:
    hy = importlib.import_module("flag_gems.runtime.backend._hygon.ops.rrelu_with_noise")
except ImportError:  # a different backend is active; load the file directly
    import importlib.util

    _path = (
        "/workspace/FlagGems/src/flag_gems/runtime/backend/_hygon/"
        "ops/rrelu_with_noise.py"
    )
    _spec = importlib.util.spec_from_file_location("hygon_rrelu", _path)
    hy = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(hy)

N = 4096
LOWER, UPPER = 0.125, 0.3333333333333333

x = torch.randn(N, device=flag_gems.device)
noise = torch.zeros_like(x)
out = torch.empty_like(x)
# 256 MiB, the same flush the benchmark's do_bench uses, so "in flight" here
# means the same thing it means there.
flush_buf = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=flag_gems.device)

gen = torch_device_fn.default_generators[torch_device_fn.current_device()]
state = gen.get_state()
FLUSH_US = None


def bench(fn):
    """Median wall time of one call under ``do_bench``, in microseconds.

    The last two signatures are there for Triton forks whose ``do_bench``
    does not take ``return_mode``; a row is only reported broken if all
    three fail.
    """
    fatal = None
    for kwargs in (
        {"warmup": 25, "rep": 100, "return_mode": "median"},
        {"warmup": 25, "rep": 100},
        {},
    ):
        try:
            return f"{triton.testing.do_bench(fn, **kwargs) * 1e3:9.1f}"
        except TypeError as exc:
            fatal = exc
            continue
        except Exception as exc:  # noqa: BLE001 - diagnostic
            return f"  FAILED({type(exc).__name__}: {exc})"
    return f"  FAILED({type(fatal).__name__}: {fatal})"


def cpu_cost(fn, n=500, warm=50):
    """CPU enqueue cost per call, no sync inside the loop."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    torch.cuda.synchronize()
    return elapsed


def blocked_cost(fn, k=25):
    """CPU time to return from ``fn`` right after a large kernel is queued.

    The kernel below keeps the device busy well past any Python cost, so a
    result near zero means ``fn`` never waits on the device and a result near
    FLUSH_US means it drains it.  ``torch.cuda.synchronize()`` at the top of
    each round keeps the queue from growing across rounds.
    """
    samples = []
    for round_ in range(k + 2):  # first two rounds warm the allocator up
        torch.cuda.synchronize()
        flush_buf.zero_()
        start = time.perf_counter()
        fn()
        if round_ >= 2:
            samples.append((time.perf_counter() - start) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(samples)


# A fixed seed/offset lets the training launch be measured without the
# generator round trip, i.e. what the floor becomes once that cost is gone.
FIXED = (1234, 0)


def train_launch_no_seed():
    n_elements = out.numel()
    block = hy._UNIFORM_HEURISTICS["BLOCK"]({"N": n_elements})
    num_warps = hy._UNIFORM_HEURISTICS["num_warps"]({"N": n_elements})
    grid = (triton.cdiv(n_elements, block * hy._TRAIN_UNROLL),)
    with torch_device_fn.device(x.device):
        hy._rrelu_with_noise_train_contiguous_kernel[grid](
            x,
            noise,
            out,
            n_elements,
            LOWER,
            UPPER,
            FIXED[0],
            FIXED[1],
            BLOCK=block,
            num_warps=num_warps,
        )


ROWS = [
    ("aten eval", lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
    ("aten train", lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("empty_like", lambda: torch.empty_like(x, memory_format=torch.contiguous_format)),
    ("philox_seed_offset", lambda: philox_backend_seed_offset(N // 4)),
    ("get_state", lambda: gen.get_state()),
    ("set_state", lambda: gen.set_state(state)),
    ("unpack tolist", lambda: state.view(torch.int64).tolist()),
    ("launch eval", lambda: hy._launch_contiguous_eval(x, out, 0.2292)),
    ("launch train", lambda: hy._launch_contiguous_train(x, noise, out, LOWER, UPPER, None)),
    ("launch train, seed given", train_launch_no_seed),
    ("hygon eval", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
    ("hygon train", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("hygon train inplace", lambda: hy.rrelu_with_noise_(x, noise, LOWER, UPPER, True)),
]

print(f"device            {torch.cuda.get_device_name(0)}")
print(f"vendor            {flag_gems.vendor_name}")
print(f"fused kernel      {hasattr(hy, '_rrelu_with_noise_train_group')}")
print(f"source            {hy.__file__}")
print(f"generator         {type(gen).__name__}, state on {state.device}")
print(f"gems device       {flag_gems.device}")

torch.cuda.synchronize()
flush_buf.zero_()
torch.cuda.synchronize()
start = time.perf_counter()
flush_buf.zero_()
torch.cuda.synchronize()
FLUSH_US = (time.perf_counter() - start) * 1e6
print(f"flush kernel      {FLUSH_US:9.1f} us on device (256 MiB memset)")

print("\n                   do_bench(us)  cpu_enqueue(us)")
for name, fn in ROWS:
    print(f"{name:22s} {bench(fn)}   {cpu_cost(fn):7.1f}")

print("\n                   do_bench(us)  cpu_enqueue(us)")
with flag_gems.use_gems(exclude=["zero_"]):
    for name, fn in ROWS[1:]:
        print(f"{name:22s} {bench(fn)}   {cpu_cost(fn):7.1f}")

print(f"\nsection 2: CPU time to return with a {FLUSH_US:.0f} us kernel queued")
print("(control row enqueues nothing; ~flush time means the piece waits on the device)")
print("\n                   blocked(us)")
print(f"{'-':22s} {blocked_cost(lambda: None):9.1f}")
with flag_gems.use_gems(exclude=["zero_"]):
    for name, fn in ROWS:
        print(f"{name:22s} {blocked_cost(fn):9.1f}")
