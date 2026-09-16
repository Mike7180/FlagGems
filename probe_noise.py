# Temporary diagnostic script -- not part of the operator change.
#
# Splits one training-mode rrelu_with_noise call into its parts and times each
# one, to find where the ~130 us of per-call overhead on this backend sits.
#
#   python probe_noise.py
#
# Run it from the repository root, with the Hygon backend active.
import time

import torch

import flag_gems
from flag_gems.runtime.backend._hygon.ops.rrelu_with_noise import (
    _fill_uniform_contiguous as fill,
)
from flag_gems.runtime.backend._hygon.ops.rrelu_with_noise import (
    _launch_contiguous_train as kernel,
)
from flag_gems.utils.random_utils import philox_backend_seed_offset as philox

buf = torch.zeros(4096, dtype=torch.float16, device="cuda")


def t(fn, n=300):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return round((time.perf_counter() - start) / n * 1e6, 1)


with flag_gems.use_gems():
    print("loop-overhead", t(lambda: None), "us")
    print("philox       ", t(lambda: philox(1024)), "us")
    print("unif-generic ", t(lambda: buf.uniform_(0.125, 0.3333)), "us")
    print("fill-new     ", t(lambda: fill(buf, 0.125, 0.3333, None)), "us")
    print("train-kernel ", t(lambda: kernel(buf, buf, buf)), "us")

print()
print("Reference: small-shape eval latency is ~7 us, small-shape train ~134 us.")
print("fill-new minus train-kernel is the real cost of generating the noise.")
