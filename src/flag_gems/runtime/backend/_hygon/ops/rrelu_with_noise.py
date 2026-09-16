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

"""Hygon-tuned ``rrelu_with_noise``.

The generic implementation routes every call through ``pointwise_dynamic``.
On this backend the per-call wrapper cost is large compared with a plain
kernel launch, and for small inputs it dominates the measured latency: the
kernel itself needs a few microseconds while the wrapper adds on the order of
a hundred.  This file serves contiguous inputs with a direct launch (the same
shape of fast path the Hygon gelu kernels use) and keeps ``pointwise_dynamic``
as the fallback for strided inputs and for tensors too large for int32
offsets.

This module is registered by the Hygon ``SpecOpRegistrar``, which overrides
the generic ``rrelu_with_noise`` / ``rrelu_with_noise_`` by function name, so
no other backend is affected.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333

_CONTIGUOUS_BLOCK_SIZE = 2048
_CONTIGUOUS_NUM_WARPS = 8
_INT32_MAX = torch.iinfo(torch.int32).max


@triton.jit
def _rrelu_with_noise_eval_contiguous_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, tl.where(x > 0, x, x * slope), mask=mask)


@triton.jit
def _rrelu_with_noise_train_contiguous_kernel(
    x_ptr,
    noise_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # ATen samples for self <= 0 (including signed zero), and records one for
    # positive/NaN elements.  Keeping this predicate aligned with backward is
    # important because noise is the training-time gradient multiplier.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    noise = tl.load(noise_ptr + offsets, mask=mask, other=0)

    not_positive = x <= 0
    effective_noise = tl.where(not_positive, noise, 1.0)
    output = tl.where(not_positive, x * effective_noise, x)

    tl.store(out_ptr + offsets, output, mask=mask)
    tl.store(noise_ptr + offsets, effective_noise, mask=mask)


# Fallback paths.  These are the same pointwise_dynamic kernels the generic
# implementation uses; they are defined here so this module stays
# self-contained.
@pointwise_dynamic(
    is_tensor=[True, False],
    num_outputs=1,
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_eval_generic(self, slope):
    return tl.where(self > 0, self, self * slope)


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_train_generic(self, noise):
    not_positive = self <= 0
    effective_noise = tl.where(not_positive, noise, 1.0)
    output = tl.where(not_positive, self * effective_noise, self)
    return output, effective_noise


def _can_use_contiguous_path(self, noise):
    return self.is_contiguous() and noise.is_contiguous() and self.numel() <= _INT32_MAX


def _launch_contiguous_eval(self, out, slope):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, _CONTIGUOUS_BLOCK_SIZE),)
    with torch_device_fn.device(self.device):
        _rrelu_with_noise_eval_contiguous_kernel[grid](
            self,
            out,
            n_elements,
            slope,
            BLOCK_SIZE=_CONTIGUOUS_BLOCK_SIZE,
            num_warps=_CONTIGUOUS_NUM_WARPS,
        )
    return out


def _launch_contiguous_train(self, noise, out):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, _CONTIGUOUS_BLOCK_SIZE),)
    with torch_device_fn.device(self.device):
        _rrelu_with_noise_train_contiguous_kernel[grid](
            self,
            noise,
            out,
            n_elements,
            BLOCK_SIZE=_CONTIGUOUS_BLOCK_SIZE,
            num_warps=_CONTIGUOUS_NUM_WARPS,
        )
    return out


def _check_rrelu_with_noise_args(self, noise, lower, upper):
    if self.shape != noise.shape:
        raise RuntimeError(
            "noise tensor must have the same shape as self. "
            f"Got self.shape = {tuple(self.shape)} "
            f"and noise.shape = {tuple(noise.shape)}"
        )
    if self.device != noise.device:
        raise RuntimeError(
            f"self and noise must be on the same device, got "
            f"{self.device} and {noise.device}"
        )
    if self.dtype != noise.dtype:
        raise RuntimeError(
            f"self and noise must have the same dtype, got "
            f"{self.dtype} and {noise.dtype}"
        )
    if not self.is_floating_point():
        raise RuntimeError(
            f"rrelu_with_noise is not implemented for dtype {self.dtype}"
        )
    if not math.isfinite(float(lower)):
        raise RuntimeError(f"rrelu: lower bound must be finite, got {lower}")
    if not math.isfinite(float(upper)):
        raise RuntimeError(f"rrelu: upper bound must be finite, got {upper}")
    if float(lower) > float(upper):
        raise RuntimeError(
            f"Lower bound should be less than or equal to the upper bound, "
            f"got lower={lower} and upper={upper}"
        )


def _fill_training_noise(noise, lower, upper, generator):
    # For a strided workspace, sample contiguously and let the training kernel
    # scatter effective noise into the caller's layout while producing output.
    if noise.is_contiguous():
        noise.uniform_(float(lower), float(upper), generator=generator)
        return noise

    sampled = torch.empty_like(noise, memory_format=torch.contiguous_format)
    sampled.uniform_(float(lower), float(upper), generator=generator)
    return sampled


def _rrelu_with_noise_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        return torch.empty_like(self) if out is None else out

    # `out` is either None (allocate) or `self` (in-place variant); anything
    # else is not reachable through the public API and takes the generic path.
    inplace = out is not None and out is self
    allocate = out is None
    fast_path = (inplace or allocate) and _can_use_contiguous_path(self, noise)

    if not training:
        slope = (float(lower) + float(upper)) * 0.5
        if fast_path:
            return _launch_contiguous_eval(
                self, self if inplace else torch.empty_like(self), slope
            )
        if allocate:
            return _rrelu_with_noise_eval_generic(self, slope)
        return _rrelu_with_noise_eval_generic(self, slope, out0=out)

    sampled_noise = _fill_training_noise(noise, lower, upper, generator)
    if fast_path and sampled_noise is noise:
        return _launch_contiguous_train(
            self, noise, self if inplace else torch.empty_like(self)
        )

    if allocate:
        output, _ = _rrelu_with_noise_train_generic(self, sampled_noise, out1=noise)
        return output
    _rrelu_with_noise_train_generic(self, sampled_noise, out0=out, out1=noise)
    return out


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise (Hygon backend)."""
    logger.debug("GEMS_HYGON RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise_ (Hygon backend)."""
    logger.debug("GEMS_HYGON RRELU_WITH_NOISE_")
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator, out=self
    )


__all__ = ["rrelu_with_noise", "rrelu_with_noise_"]
