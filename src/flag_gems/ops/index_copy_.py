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

import importlib
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_copy_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # Decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # Signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("inp,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("K: tl.constexpr,")
            code.writeline("DO_COPY: tl.constexpr,")
            code.writeline("inp_numel: tl.constexpr,")
            code.writeline("inp_stride_dim: tl.constexpr,")
            code.writeline("inp_shape_dim: tl.constexpr,")
            code.writeline("src_shape_dim: tl.constexpr,")
            code.writeline("delta: tl.constexpr,")

            stride_args = ", ".join(
                f"src_stride_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"src_shape_{i}: tl.constexpr" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_K: tl.constexpr,")
            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

        # Kernel
        with code.indent():
            code.writeline("pid = tl.program_id(axis=0)")
            code.writeline("blk = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("offsets = blk")
            code.writeline("mask = offsets < N")

            for i in range(rank - 1, -1, -1):
                code.writeline(f"src_offset{i} = offsets % src_shape_{i}")
                code.writeline(f"offsets = offsets // src_shape_{i}")
            code.newline()
            comp = [f"src_offset{i} * src_stride_{i}" for i in range(rank)]
            code.writeline(f"src_offset = {' + '.join(comp)}")

            code.writeline("pre_cal = inp_stride_dim * src_shape_dim")

            # index copy
            code.writeline("pre_idx = (src_offset // pre_cal).to(tl.int64)")
            code.writeline(
                "dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)"
            )
            code.writeline(
                "src_dim_idx = tl.load(index + dim_idx, mask=mask, other=0).to(tl.int64)"
            )
            code.writeline(
                "valid_index = (src_dim_idx >= 0) & (src_dim_idx < inp_shape_dim)"
            )
            code.writeline(
                "tl.device_assert((~mask) | valid_index, "
                '"index value out of bounds: 0 <= index < self.size(dim)")'
            )

            code.writeline(
                "input_idx = (src_offset + "
                "(delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim"
                ").to(tl.int64)"
            )

            code.writeline("input_mask = (input_idx >= 0) & (input_idx < inp_numel)")
            code.writeline("store_mask = mask & valid_index & input_mask")
            code.writeline("src_val = tl.load(src + src_offset, mask=mask, other=0)")
            code.writeline("tl.store(out + input_idx, src_val, mask=store_mask)")

            # Fused clone: copy inp into out in the same launch, skipping the
            # positions the scatter above already owns. The two writes are
            # disjoint by construction, so their order does not matter.
            #
            # A lane's dim coordinate is (o // inp_stride_dim) % inp_shape_dim.
            # The caller only enables DO_COPY when inp_stride_dim >= BLOCK_SIZE,
            # so a block covers at most two consecutive dim coordinates: the
            # first `split` lanes sit in `dim_base`'s period, the rest in the
            # next one. Two scalar membership tests therefore cover the block.
            code.writeline("if DO_COPY:")
            with code.indent():
                code.writeline("cmask = blk < inp_numel")
                code.writeline(
                    "dim_base = "
                    "((pid * BLOCK_SIZE) // inp_stride_dim) % inp_shape_dim"
                )
                code.writeline("koffs = tl.arange(0, BLOCK_K)")
                code.writeline(
                    "kidx = tl.load(index + koffs, mask=koffs < K, other=-1)"
                    ".to(tl.int64)"
                )
                code.writeline(
                    "cov_base = tl.max("
                    "(kidx == dim_base.to(tl.int64)).to(tl.int32), axis=0)"
                )
                code.writeline("dim_next = (dim_base + 1) % inp_shape_dim")
                code.writeline(
                    "cov_next = tl.max("
                    "(kidx == dim_next.to(tl.int64)).to(tl.int32), axis=0)"
                )
                code.writeline(
                    "split = inp_stride_dim - ((pid * BLOCK_SIZE) % inp_stride_dim)"
                )
                code.writeline(
                    "covered = tl.where("
                    "tl.arange(0, BLOCK_SIZE) >= split, cov_next, cov_base)"
                )
                code.writeline("copy_mask = cmask & (covered == 0)")
                code.writeline("copy_val = tl.load(inp + blk, mask=copy_mask, other=0)")
                code.writeline("tl.store(out + blk, copy_val, mask=copy_mask)")

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, src, inp, dim, inp_stride_dim, inp_shape_dim, src_shape_dim,
    # delta, N, inp.numel(), do_copy, block_size
    parameters: List[str] = []
    parameters.append("out")
    parameters.append("index")
    parameters.append("src")
    parameters.append("inp")
    parameters.append("dim")
    parameters.append("inp_stride_dim")
    parameters.append("inp_shape_dim")
    parameters.append("src_shape_dim")
    parameters.append("delta")
    parameters.append("N")
    parameters.append("inp_numel")
    parameters.append("do_copy")
    parameters.append("block_size")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")

        # Kernel launch
        code.writeline("if N <= 4096:")
        code.writeline("    BLOCK_SIZE = 64")
        code.writeline("elif N <= 65536:")
        code.writeline("    BLOCK_SIZE = 128")
        code.writeline("elif N <= 524288:")
        code.writeline("    BLOCK_SIZE = 256")
        code.writeline("else:")
        code.writeline("    BLOCK_SIZE = 512")
        # The fused launch also covers inp, so its grid is sized by whichever
        # of the two is larger. The plain launch only scatters src.
        code.writeline("if do_copy:")
        with code.indent():
            code.writeline("K = index.numel()")
            code.writeline("BLOCK_SIZE = block_size")
            code.writeline("BLOCK_K = triton.next_power_of_2(K)")
            code.writeline("grid = (triton.cdiv(max(N, inp_numel), BLOCK_SIZE),)")
        code.writeline("else:")
        with code.indent():
            code.writeline("K = 1")
            code.writeline("BLOCK_K = 1")
            code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, src, inp, out, N, K, do_copy, inp_numel, inp_stride_dim, "
                "inp_shape_dim, src_shape_dim, delta, "
            )
            if rank > 0:
                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_K=BLOCK_K,")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, src, inp, dim, inp_stride_dim, inp_shape_dim,
    #          src_shape_dim, delta, N, inp.numel(), do_copy, block_size]
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_index_copy_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexCopyFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Dict[int, Callable[..., Any]] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        if key in self.overloads:
            return self.overloads[key](*args, **kwargs)

        code = IndentedBuffer()
        code = generate_code(
            args,
            "_index_copy_wrapper",
            "_index_copy_jit_function",
            code,
        )

        file_name = f"index_copy_rank_{key}_pid_{self.pid}.py"

        try:
            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_copy_wrapper")
            self.overloads[key] = overload
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate or load index_copy kernel: {e}"
            ) from e

        return overload(*args, **kwargs)

    def arg_key(self, *args) -> int:
        # Cache per rank: shape and stride are passed to Triton as
        # tl.constexpr, so Triton specializes the kernel on its own.
        src = args[2]
        return src.ndim


_index_copy_func = IndexCopyFunction()


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def _index_copy_clone_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


def _clone_without_copy_dispatch(inp):
    # Lightweight Triton copy to avoid dispatch interference with FlagGems' copy_ override.
    if not inp.is_contiguous():
        return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)

    out = torch.empty_like(inp)
    n_elements = inp.numel()
    if n_elements == 0:
        return out

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _index_copy_clone_kernel[grid](
        inp,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out


# Fusing trades the extra work of one merged kernel for one saved launch, so it
# only pays off while the payload is small. On the H20-3e the two paths are
# even at ~16 MiB and fusing loses past that; stop at 8 MiB to stay clear. A
# device whose launch is more expensive than the H20's can afford a larger cap.
_MAX_FUSED_BYTES = 8 * 1024 * 1024

# The fused kernel scans the whole index once per block, so the index length is
# held to a small multiple of the block it guards.
_MAX_SCAN_RATIO = 2


def _fused_block_size(inp, index, inp_stride_dim, N):
    """Pick the block size for a fused clone+scatter launch, or 0 to not fuse.

    The fused kernel resolves a block's dim coordinates from two scalars, which
    is exact only while a block spans at most two dim-stride periods. Shrink the
    block until the stride can hold it, and give up if it gets too small to be
    worth launching.

    Note the innermost dimension never qualifies: its stride is 1, so a block
    would need a per-lane membership test.
    """
    if not inp.is_contiguous() or inp_stride_dim <= 0:
        return 0
    n_index = index.numel()
    if N == 0 or n_index == 0:
        return 0
    if inp.numel() * inp.element_size() > _MAX_FUSED_BYTES:
        return 0

    n_elements = inp.numel()
    if n_elements <= 4096:
        block_size = 64
    elif n_elements <= 65536:
        block_size = 128
    elif n_elements <= 524288:
        block_size = 256
    else:
        block_size = 512

    while block_size > inp_stride_dim:
        block_size //= 2
    if block_size < 64 or n_index > _MAX_SCAN_RATIO * block_size:
        return 0
    return block_size


def index_copy(inp, dim, index, src):
    logger.debug("GEMS INDEX_COPY")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()
    inp_numel = inp.numel()

    fused_block_size = _fused_block_size(inp, index, inp_stride_dim, N)

    with torch_device_fn.device(inp.device):
        if fused_block_size:
            # One launch does both the clone and the scatter: out starts
            # uninitialised, the kernel copies inp into it and overwrites the
            # indexed positions with src.
            out = torch.empty_like(inp)
            _index_copy_func(
                out,
                index,
                src,
                inp,
                dim,
                inp_stride_dim,
                inp_shape_dim,
                src_shape_dim,
                delta,
                N,
                inp_numel,
                True,
                fused_block_size,
            )
            return out

        # inp is not contiguous, or the index is large enough that scanning it
        # per block costs more than a second launch: clone first, then scatter.
        out = _clone_without_copy_dispatch(inp)
        if N > 0:
            _index_copy_func(
                out,
                index,
                src,
                inp,
                dim,
                inp_stride_dim,
                inp_shape_dim,
                src_shape_dim,
                delta,
                N,
                inp_numel,
                False,
                0,
            )
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS INDEX_COPY_")
    assert -inp.ndim <= dim < inp.ndim, "Invalid dim"
    dim %= inp.ndim
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    if N > 0:
        with torch_device_fn.device(inp.device):
            # In-place: there is nothing to clone, so the fused copy is off and
            # inp is passed only to keep the wrapper signature uniform.
            _index_copy_func(
                inp,
                index,
                src,
                inp,
                dim,
                inp_stride_dim,
                inp_shape_dim,
                src_shape_dim,
                delta,
                N,
                inp.numel(),
                False,
                0,
            )
    return inp
