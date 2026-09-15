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

"""Unit tests for ``flag_gems.rrelu_with_noise`` and ``flag_gems.rrelu_with_noise_``.

Both operators implement::

    aten::rrelu_with_noise(Tensor self, Tensor(a!) noise, Scalar lower,
                           Scalar upper, bool training, Generator? generator)

whose contract is:

* ``training=False``: ``out = self > 0 ? self : self * (lower + upper) / 2`` and
  the caller's ``noise`` buffer is left untouched;
* ``training=True``: every ``self <= 0`` element gets a slope drawn uniformly
  from ``[lower, upper]`` and stored in ``noise``, every other element stores
  ``1``, and ``out = self * noise``.

The tests either compare the FlagGems API against the native ATen kernel (the
unit-test reference, run on ``utils.to_reference`` tensors) or verify an
invariant of the contract above that no reference is needed for.  The FlagGems
API is always called explicitly and without ``flag_gems.use_gems``, as required
by ``docs/content/zh-cn/contribution/overview.md``.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Defaults of torch.nn.functional.rrelu.
DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 1.0 / 3.0

# A training call draws its slopes from the global generator, which cannot be
# replayed on the reference side.  Equal bounds remove that randomness -- every
# sampled element then holds exactly this slope -- so that the applied noise and
# the output stay comparable element-wise with ATen.  The sampling itself is
# covered by test_rrelu_with_noise_train_sampling.
EQUAL_BOUNDS = (0.25, 0.25)

# Values that pick a branch of the kernels: the training slope is drawn for
# `self <= 0` (signed zero included), the eval slope for `self <= 0` as well,
# and NaN never takes the `self > 0` side.
BOUNDARY_VALUES = [
    float("nan"),
    float("inf"),
    float("-inf"),
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
]

OP_NAMES = ["rrelu_with_noise", "rrelu_with_noise_"]

# The implementation under test, called through the public FlagGems API.
GEMS_OP = {
    "rrelu_with_noise": flag_gems.rrelu_with_noise,
    "rrelu_with_noise_": flag_gems.rrelu_with_noise_,
}

# The native operator, used as the unit-test reference.
ATEN_OP = {
    "rrelu_with_noise": torch.ops.aten.rrelu_with_noise,
    "rrelu_with_noise_": torch.ops.aten.rrelu_with_noise_,
}


def _run(op_name, side, self, noise, lower, upper, training, generator=None):
    """Call one side (FlagGems or ATen) with the operator's public signature."""
    return side[op_name](self, noise, lower, upper, training, generator)


def _skip_half_cpu_reference(dtype, training):
    """Skip the cases the CPU reference cannot serve.

    ``torch.ops.aten.rrelu_with_noise`` has no Half kernel for the training
    path on CPU: it raises "rrelu_with_noise_out_cpu not implemented for
    'Half'".  fp16 keeps being covered by the reference-free tests.
    """
    if cfg.TO_CPU and training and dtype == torch.float16:
        pytest.skip("the CPU reference does not implement the training path for Half")


def _pair(shape, dtype, bounds, fill_noise=False):
    """Build matched (FlagGems, reference) inputs.

    ``fill_noise`` pre-fills the caller's noise buffer with a random slope, so
    that the eval path is tested against a buffer that differs from the unit
    slope rather than against zeros.
    """
    lower, upper = bounds
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if fill_noise:
        noise.uniform_(lower, upper)
    return (
        inp,
        noise,
        utils.to_reference(inp.clone()),
        utils.to_reference(noise.clone()),
    )


def _forward_backward(side, op_name, base, bounds, training):
    """One forward/backward pass through ``side``; returns (out, noise, grad).

    The gradient is read from ``leaf``, the only tensor of the graph that
    requires grad, so that both sides can be compared through it.
    """
    leaf = base.clone().requires_grad_()
    if op_name.endswith("_"):
        # An in-place operator rejects a leaf that requires grad; a non-leaf
        # self is the legal path and exercises the self_is_result branch.
        self_t = leaf * 1.0
    else:
        self_t = leaf
    noise = torch.zeros_like(self_t)
    out = _run(op_name, side, self_t, noise, *bounds, training)
    out.sum().backward()
    return out, noise, leaf.grad


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_eval(op_name, shape, dtype):
    """Eval mode matches ATen and leaves the caller's noise buffer untouched."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    inp, noise, ref_inp, ref_noise = _pair(
        shape, dtype, (lower, upper), fill_noise=True
    )

    ref_out = _run(op_name, ATEN_OP, ref_inp, ref_noise, lower, upper, False)
    result = _run(op_name, GEMS_OP, inp, noise, lower, upper, False)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_train(op_name, shape, dtype):
    """Training mode matches ATen: the same effective noise and output.

    The bounds are equal so that the sampled slope is deterministic; see
    ``EQUAL_BOUNDS``.
    """
    _skip_half_cpu_reference(dtype, True)
    lower, upper = EQUAL_BOUNDS
    inp, noise, ref_inp, ref_noise = _pair(shape, dtype, (lower, upper))

    ref_out = _run(op_name, ATEN_OP, ref_inp, ref_noise, lower, upper, True)
    result = _run(op_name, GEMS_OP, inp, noise, lower, upper, True)

    utils.gems_assert_close(result, ref_out, dtype)
    # `noise` reports the effective slope: the sampled one for `self <= 0` and
    # a unit slope everywhere else.
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_rrelu_with_noise_train_sampling(op_name, dtype):
    """The slope must be uniform over the whole ``[lower, upper]`` range and be
    drawn for exactly the ``self <= 0`` elements."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    span = upper - lower
    original = torch.linspace(-2.0, 2.0, 4097, dtype=dtype, device=flag_gems.device)
    inp = original.clone()
    noise = torch.zeros_like(inp)

    result = _run(op_name, GEMS_OP, inp, noise, lower, upper, True)

    sampled = original <= 0
    assert 0 < int(sampled.sum()) < sampled.numel()
    drawn = noise[sampled]

    assert torch.all(drawn >= lower)
    assert torch.all(drawn <= upper)
    # ~2000 draws must cover the interval rather than a sub-range of it: their
    # extreme values sit close to the bounds and their mean close to the middle.
    assert drawn.min() <= lower + span * 1e-2
    assert drawn.max() >= upper - span * 1e-2
    assert abs(float(drawn.mean()) - (lower + upper) / 2) <= span * 1e-1
    assert torch.any(drawn != drawn[0])

    # Every other element stores a unit slope, and the output is the input
    # scaled by the effective noise.
    utils.gems_assert_equal(
        noise[~sampled], utils.to_reference(torch.ones_like(noise[~sampled]))
    )
    expected = utils.to_reference(torch.where(sampled, original * noise, original))
    utils.gems_assert_close(result, expected, dtype)
    if not op_name.endswith("_"):
        utils.gems_assert_equal(inp, utils.to_reference(original))


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
def test_rrelu_with_noise_generator(op_name):
    """The generator argument drives the training slope: equal seeds reproduce
    it exactly, different seeds do not, and the stream keeps advancing."""
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER
    original = -torch.ones((4096,), dtype=torch.float32, device=flag_gems.device)

    def draw(seed):
        generator = torch.Generator(device=flag_gems.device)
        generator.manual_seed(seed)
        inp = original.clone()
        noise = torch.zeros_like(inp)
        out = _run(op_name, GEMS_OP, inp, noise, lower, upper, True, generator)
        return generator, out, noise

    _, out_a, noise_a = draw(2026)
    generator_b, out_b, noise_b = draw(2026)
    _, _, noise_c = draw(2027)

    assert torch.equal(out_a, out_b)
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_c)

    inp = original.clone()
    noise = torch.zeros_like(inp)
    _run(op_name, GEMS_OP, inp, noise, lower, upper, True, generator_b)
    assert not torch.equal(noise, noise_a)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
def test_rrelu_with_noise_eval_is_side_effect_free(op_name):
    """Eval mode consumes no randomness and writes to neither input tensor."""
    generator = torch.Generator(device=flag_gems.device)
    generator.manual_seed(2026)
    state_before = generator.get_state().clone()
    inp = torch.randn((257,), device=flag_gems.device)
    noise = torch.randn_like(inp)
    noise_before = noise.clone()
    input_before = inp.clone()

    _run(op_name, GEMS_OP, inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, False, generator)

    assert torch.equal(generator.get_state(), state_before)
    assert torch.equal(noise, noise_before)
    if not op_name.endswith("_"):
        assert torch.equal(inp, input_before)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_aliasing(op_name, training):
    """The in-place variant writes into and returns ``self``; the out-of-place
    variant returns a buffer that aliases neither ``self`` nor ``noise``."""
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    inp = torch.randn((37, 11), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp) if training else torch.rand_like(inp)
    input_ptr = inp.data_ptr()
    noise_ptr = noise.data_ptr()
    input_before = inp.clone()

    result = _run(op_name, GEMS_OP, inp, noise, lower, upper, training)

    if op_name.endswith("_"):
        assert result.data_ptr() == input_ptr
    else:
        assert result.data_ptr() != input_ptr
        assert result.data_ptr() != noise_ptr
        assert torch.equal(inp, input_before)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", [(0,), (0, 7), (2, 0, 3)])
def test_rrelu_with_noise_empty(op_name, training, shape):
    """Empty inputs keep the shape, the dtype and the in-place contract."""
    inp = torch.empty(shape, device=flag_gems.device)
    noise = torch.empty_like(inp)
    input_ptr = inp.data_ptr()

    result = _run(op_name, GEMS_OP, inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, training)

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    if op_name.endswith("_"):
        assert result.data_ptr() == input_ptr
    # The out-of-place aliasing contract is not asserted here: zero-sized
    # buffers all share one address, so neither the pointer nor the object
    # identity tells an allocated result from the input.  See
    # test_rrelu_with_noise_aliasing for that contract.


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_rrelu_with_noise_non_contiguous(op_name, training, dtype):
    """Strided views match ATen and leave the memory outside the view alone."""
    _skip_half_cpu_reference(dtype, training)
    lower, upper = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    input_base = torch.linspace(
        -2.0, 2.0, 17 * 22, dtype=dtype, device=flag_gems.device
    ).reshape(17, 22)
    noise_base = torch.zeros_like(input_base)
    untouched_input = input_base[:, 1::2].clone()
    untouched_noise = noise_base[:, 1::2].clone()

    inp = input_base[:, ::2]
    noise = noise_base[:, ::2]
    assert not inp.is_contiguous()
    assert not noise.is_contiguous()

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = utils.to_reference(noise.clone())
    ref_out = _run(op_name, ATEN_OP, ref_inp, ref_noise, lower, upper, training)

    result = _run(op_name, GEMS_OP, inp, noise, lower, upper, training)

    utils.gems_assert_close(result, ref_out, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)
    assert torch.equal(input_base[:, 1::2], untouched_input)
    assert torch.equal(noise_base[:, 1::2], untouched_noise)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_autograd(op_name, training):
    """Gradients match ATen in both training modes."""
    bounds = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    base = torch.randn((257,), dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    _, _, ref_grad = _forward_backward(ATEN_OP, op_name, ref_base, bounds, training)
    _, _, gems_grad = _forward_backward(GEMS_OP, op_name, base, bounds, training)

    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_autograd_boundary(op_name, training):
    """Signed zero, infinities and NaN take the same branch as ATen in the
    forward and in the backward kernel.

    The kernels select their slope with ``self > 0``, so an implementation that
    tests ``self >= 0`` would diverge exactly on these values.
    """
    bounds = EQUAL_BOUNDS if training else (DEFAULT_LOWER, DEFAULT_UPPER)
    base = torch.tensor(BOUNDARY_VALUES, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    ref_out, ref_noise, ref_grad = _forward_backward(
        ATEN_OP, op_name, ref_base, bounds, training
    )
    gems_out, gems_noise, gems_grad = _forward_backward(
        GEMS_OP, op_name, base, bounds, training
    )

    utils.gems_assert_close(gems_out, ref_out, torch.float32, equal_nan=True)
    utils.gems_assert_close(gems_noise, ref_noise, torch.float32)
    utils.gems_assert_close(gems_grad, ref_grad, torch.float32)


@pytest.mark.rrelu_with_noise
@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("op_name", OP_NAMES)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "case,match",
    [
        ("noise_shape", "same shape"),
        ("noise_dtype", "same dtype"),
        ("self_dtype", "not implemented"),
        ("noise_device", "same device"),
        ("lower_infinite", "must be finite"),
        ("upper_nan", "must be finite"),
        ("bounds_swapped", "less than or equal"),
    ],
)
def test_rrelu_with_noise_invalid_args(op_name, training, case, match):
    """Invalid arguments are rejected before either tensor is touched."""
    inp = torch.randn((5,), dtype=torch.float32, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    lower, upper = DEFAULT_LOWER, DEFAULT_UPPER

    if case == "noise_shape":
        noise = torch.zeros((3,), dtype=torch.float32, device=flag_gems.device)
    elif case == "noise_dtype":
        noise = torch.zeros((5,), dtype=torch.int32, device=flag_gems.device)
    elif case == "self_dtype":
        inp = torch.ones((5,), dtype=torch.int32, device=flag_gems.device)
        noise = torch.zeros((5,), dtype=torch.int32, device=flag_gems.device)
    elif case == "noise_device":
        if flag_gems.device == "cpu":
            pytest.skip("needs a second device to pair with the CPU reference")
        noise = torch.zeros((5,), dtype=torch.float32, device="cpu")
    elif case == "lower_infinite":
        lower = float("inf")
    elif case == "upper_nan":
        upper = float("nan")
    elif case == "bounds_swapped":
        lower, upper = DEFAULT_UPPER, DEFAULT_LOWER

    input_before = inp.clone()
    noise_before = noise.clone()

    with pytest.raises(RuntimeError, match=match):
        _run(op_name, GEMS_OP, inp, noise, lower, upper, training)

    assert torch.equal(inp, input_before)
    assert torch.equal(noise, noise_before)
