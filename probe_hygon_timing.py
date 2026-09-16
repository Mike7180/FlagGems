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

Temporary diagnostic, not part of the operator.  Three columns per row:

``cpu``      time to enqueue the call, with no synchronize in the loop.
``device``   device time the call adds on top of a large memset that is
             already in flight.  A piece that only burns host time adds ~0;
             one that queues a copy or a kernel adds that work's duration.
             This is what ``do_bench`` cannot separate, and it is the reason
             the training floor is ~60 us above the eval floor.
``do_bench`` the harness' own measurement, for comparison with its tables.

The ``launch train`` rows isolate the suspected cost: they all run the same
kernel, and differ only in the seed/offset arguments.

Run it on the benchmark host with ``python probe_hygon_timing.py`` and paste
the output back.
"""

import importlib
import statistics
import time

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

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
BIG = 1 << 40

x = torch.randn(N, device=flag_gems.device)
noise = torch.zeros_like(x)
out = torch.empty_like(x)
# 256 MiB, the same flush the harness' do_bench uses.
flush_buf = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=flag_gems.device)

gen = torch_device_fn.default_generators[torch_device_fn.current_device()]
state = gen.get_state()

BLOCK = hy._UNIFORM_HEURISTICS["BLOCK"]({"N": N})
NUM_WARPS = hy._UNIFORM_HEURISTICS["num_warps"]({"N": N})
GRID = (triton.cdiv(N, BLOCK * hy._TRAIN_UNROLL),)


def kernel_args(seed, offset):
    """Everything the training launch needs, without calling the launcher."""
    return (
        x,
        noise,
        out,
        N,
        float(LOWER),
        float(UPPER),
        seed,
        offset,
    )


def train_launch(seed=None, offset=None):
    if seed is None:
        seed, offset = philox_backend_seed_offset(triton.cdiv(N, hy._TRAIN_UNROLL))
    with torch_device_fn.device(x.device):
        hy._rrelu_with_noise_train_contiguous_kernel[GRID](
            *kernel_args(seed, offset), BLOCK=BLOCK, num_warps=NUM_WARPS
        )


_counter = [0]


def _empty():
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def train_launch_changing():
    _counter[0] += 1
    train_launch(BIG + _counter[0], BIG + 2 * _counter[0])


# Candidate: same kernel, but the seed/offset are loaded from a small device
# buffer instead of being launch arguments.  The launch arguments are then
# constant across calls, which is the thing under test.
@triton.jit
def _train_from_state_kernel(
    x_ptr, noise_ptr, out_ptr, state_ptr, N, lower, upper, BLOCK: tl.constexpr
):
    philox_seed = tl.load(state_ptr)
    philox_offset = tl.load(state_ptr + 1)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(axis=0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += lane
    zero = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, zero, zero)
    scale = upper - lower
    r0 = uint_to_uniform_float(r0) * scale + lower
    r1 = uint_to_uniform_float(r1) * scale + lower
    r2 = uint_to_uniform_float(r2) * scale + lower
    r3 = uint_to_uniform_float(r3) * scale + lower
    off_0 = pid * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    hy._rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_0, r0, N)
    hy._rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_1, r1, N)
    hy._rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_2, r2, N)
    hy._rrelu_with_noise_train_group(x_ptr, noise_ptr, out_ptr, off_3, r3, N)


STATE_DEV = torch.zeros(2, dtype=torch.int64, device=flag_gems.device)
STATE_HOST = torch.zeros(2, dtype=torch.int64, pin_memory=True)


def state_copy(seed=BIG + 1, offset=BIG + 2):
    """Write a new seed/offset pair into the device buffer.

    The host buffer is pinned and the copy is async, so this queues work
    instead of synchronizing the way a plain ``copy_`` does.
    """
    STATE_HOST[0] = seed
    STATE_HOST[1] = offset
    STATE_DEV.copy_(STATE_HOST, non_blocking=True)


def train_launch_from_state():
    state_copy(BIG + _counter[0], BIG + 2 * _counter[0])
    with torch_device_fn.device(x.device):
        _train_from_state_kernel[GRID](
            x, noise, out, STATE_DEV, N, LOWER, UPPER, BLOCK=BLOCK, num_warps=NUM_WARPS
        )


def bench(fn):
    try:
        done = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        return f"{done * 1e3:8.1f}"
    except Exception as exc:  # noqa: BLE001 - diagnostic
        return f"  FAILED({type(exc).__name__})"


def cpu_cost(fn, n=300, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    torch.cuda.synchronize()
    return elapsed


def device_cost(fn, k=12):
    """Device time ``fn`` adds, over a 256 MiB memset that is already queued.

    The memset keeps the device busy for longer than any host-side cost here,
    so the difference is work ``fn`` queued onto the stream.
    """

    def once(extra):
        torch.cuda.synchronize()
        start = time.perf_counter()
        flush_buf.zero_()
        extra()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1e6

    base = statistics.median([once(lambda: None) for _ in range(k + 2)])
    got = statistics.median([once(fn) for _ in range(k + 2)])
    return got - base, base


torch.cuda.synchronize()
_, FLUSH = device_cost(lambda: None)

_ROWS = [
    ("aten eval", lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
    ("aten train", lambda: torch.ops.aten.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("empty_like", _empty),
    ("philox_seed_offset", lambda: philox_backend_seed_offset(N // hy._TRAIN_UNROLL)),
    ("get_state", lambda: gen.get_state()),
    ("set_state", lambda: gen.set_state(state)),
    ("launch eval", lambda: hy._launch_contiguous_eval(x, out, 0.2292)),
    ("launch train, seeded", lambda: train_launch()),
    ("launch train, fixed small", lambda: train_launch(1234, 0)),
    ("launch train, fixed big", lambda: train_launch(BIG, BIG)),
    ("launch train, changing", train_launch_changing),
    ("philox + empty_like", lambda: (philox_backend_seed_offset(N), _empty())),
    ("launch then philox", lambda: (train_launch_changing(), philox_backend_seed_offset(N))),
    ("state copy (pinned)", lambda: state_copy()),
    ("launch train, from state", train_launch_from_state),
    ("hygon eval", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, False)),
    ("hygon train", lambda: hy.rrelu_with_noise(x, noise, LOWER, UPPER, True)),
    ("hygon train inplace", lambda: hy.rrelu_with_noise_(x, noise, LOWER, UPPER, True)),
]

print(f"device            {torch.cuda.get_device_name(0)}")
print(f"vendor            {flag_gems.vendor_name}")
print(f"torch/triton      {torch.__version__} / {triton.__version__}")
print(f"fused kernel      {hasattr(hy, '_rrelu_with_noise_train_group')}")
print(f"source            {hy.__file__}")
print(f"generator         {type(gen).__name__}, state on {state.device}")
_patched = getattr(flag_gems.ops, "rrelu_with_noise", None)
print(f"gems rrelu        {getattr(_patched, '__module__', 'not in flag_gems.ops')}")
print(f"rrelu names       {[n for n in dir(flag_gems.ops) if 'rrelu' in n]}")
print(f"flush              {FLUSH:8.1f} us of device time per round")

for patch in (None, ["zero_"]):
    torch.cuda.synchronize()
    if patch is None:
        print("\noutside use_gems()")
    else:
        print("\ninside use_gems()")

    def run():
        print("                     cpu(us)  device(us)  do_bench(us)")
        for name, fn in _ROWS:
            dev, _ = device_cost(fn)
            print(f"{name:20s} {cpu_cost(fn):8.1f}  {dev:10.1f}  {bench(fn)}")

    if patch is None:
        run()
    else:
        with flag_gems.use_gems(exclude=patch):
            run()

with flag_gems.use_gems(exclude=["zero_"]):
    print("\ndispatch: which function does the patched op call?")
    for label, args in (
        ("5 positional args", (x, noise, LOWER, UPPER, True)),
        ("6 positional args", (x, noise, LOWER, UPPER, True, None)),
    ):
        fn = lambda: torch.ops.aten.rrelu_with_noise(*args)
        print(f"  {label:18s} cpu {cpu_cost(fn):8.1f} us   do_bench {bench(fn)} us")
