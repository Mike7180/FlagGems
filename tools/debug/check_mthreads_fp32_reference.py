#!/usr/bin/env python3
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

"""Probe whether torch's on-device convolution is a valid fp32 reference on MUSA.

Without ``--ref cpu`` the cudnn_convolution tests compare the FlagGems output
against torch's own convolution on the device. MUSA has no fp64, so on that
path the reference stays in fp32 on the card. If the device convolution runs a
reduced-precision (TF32/SQMMA-like) path, the reference itself is off by about
1e-3 while the fp32 tolerance is rtol=1.3e-6 / atol=1e-4, so every fp32 case
fails even though the FlagGems kernel is accurate.

For each case this prints ``max|device - cpu_fp64|`` with the defaults, then
again with the reduced-precision knobs turned off. An fp32-exact
implementation should land around 1e-6; a TF32-like path lands around 1e-3.

Run on the MUSA machine, with the MUSA torch stack available:

    python tools/debug/check_mthreads_fp32_reference.py

To also probe the FlagGems kernel from this checkout, run from the repo root:

    PYTHONPATH=src python tools/debug/check_mthreads_fp32_reference.py
"""

import torch

try:
    import torch_musa  # noqa: F401  # registers the MUSA backend
except ImportError as exc:  # pragma: no cover - diagnostic script
    raise SystemExit(f"torch_musa is required for this probe: {exc!r}")


# The test's fp32 tolerance is atol=1e-4, rtol=1.3e-6. Printing the error in
# units of atol makes the reference's excess easy to read.
F32_ATOL = 1e-4


def _max_err(y, ref):
    return (y.cpu().double() - ref).abs().max().item()


def _make_cases():
    device = torch.device("musa")

    def rand(*shape):
        return torch.randn(*shape, device=device)

    return [
        # One of the shapes that fails without --ref cpu (fp32, groups=2).
        (
            "conv2d groups=2",
            torch.nn.functional.conv2d,
            rand(4, 4, 8, 8),
            rand(4, 2, 3, 3),
            dict(padding=1, groups=2),
        ),
        (
            "conv2d groups=1",
            torch.nn.functional.conv2d,
            rand(1, 2, 5, 5),
            rand(3, 2, 3, 3),
            dict(padding=0),
        ),
        (
            "conv2d dilated",
            torch.nn.functional.conv2d,
            rand(1, 2, 9, 9),
            rand(3, 2, 3, 3),
            dict(padding=0, dilation=2),
        ),
        (
            "conv1d groups=1",
            torch.nn.functional.conv1d,
            rand(2, 2, 8),
            rand(3, 2, 3),
            dict(padding=0),
        ),
        (
            "conv1d depthwise stride=2",
            torch.nn.functional.conv1d,
            rand(1, 4, 16),
            rand(4, 1, 3),
            dict(stride=2, groups=4),
        ),
        (
            "conv3d groups=1",
            torch.nn.functional.conv3d,
            rand(1, 2, 5, 5, 5),
            rand(3, 2, 3, 3, 3),
            dict(padding=0),
        ),
    ]


def _run_cases():
    for name, conv, x, w, kwargs in _make_cases():
        try:
            y = conv(x, w, **kwargs)
            ref = conv(x.cpu().double(), w.cpu().double(), **kwargs)
            err = _max_err(y, ref)
            print(
                f"  {name:<26} max|device - cpu_fp64| = {err:.3e}"
                f"  ({err / F32_ATOL:,.1f}x atol)"
            )
        except Exception as exc:  # pragma: no cover - diagnostic script
            # MUDNN rejects some configs outright (e.g. grouped convs with
            # allow_tf32=False); keep probing the remaining cases.
            print(f"  {name:<26} FAILED: {exc}")


def _reduced_precision_knobs():
    knobs = []
    for backend_name in ("mudnn", "cudnn"):
        backend = getattr(torch.backends, backend_name, None)
        if backend is None:
            continue
        for knob in ("allow_tf32", "allow_hf32"):
            if hasattr(backend, knob):
                knobs.append((backend_name, backend, knob))
    return knobs


def _probe_flag_gems():
    try:
        import flag_gems
    except ImportError:
        print("  flag_gems is not importable; skipped")
        return

    device = torch.device("musa")
    x = torch.randn(4, 4, 8, 8, device=device)
    w = torch.randn(4, 2, 3, 3, device=device)
    try:
        y = flag_gems.cudnn_convolution(
            x,
            w,
            padding=[1, 1],
            stride=[1, 1],
            dilation=[1, 1],
            groups=2,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"  flag_gems.cudnn_convolution failed: {exc!r}")
        return
    ref = torch.nn.functional.conv2d(
        x.cpu().double(), w.cpu().double(), padding=1, groups=2
    )
    err = _max_err(y, ref)
    print(
        "  flag_gems.cudnn_convolution groups=2"
        f" max|gems - cpu_fp64| = {err:.3e}"
        f"  ({err / F32_ATOL:,.2f}x atol)"
    )


def _probe_tf32_truncation():
    # 1 + 3*2^-13 is exact in fp32 (13 significand bits) but rounds to 1.0 in
    # TF32: TF32 keeps 10 stored mantissa bits, so half an ulp at 1.0 is 2^-11
    # and 3*2^-13 < 2^-11. With an all-ones kernel the interior outputs sum
    # 2 channels x 9 taps = 18 terms, so an fp32-exact convolution returns
    # 18.006591796875 while a TF32 one returns exactly 18.0.
    device = torch.device("musa")
    value = 1.0 + 3.0 * 2.0**-13
    x = torch.full((4, 4, 8, 8), value, device=device)
    w = torch.ones(4, 2, 3, 3, device=device)
    try:
        y = torch.nn.functional.conv2d(x, w, padding=1, groups=2)
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"  conv2d groups=2 failed: {exc!r}")
        return
    interior = (x.shape[2] - 2) * (x.shape[3] - 2) * y.shape[0] * y.shape[1]
    hits = int((y == 18.0).sum().item())
    print(f"  input value = 0x{value.hex()} ({value!r})")
    print(f"  fp32-exact interior value = {18.0 * value!r}; TF32-truncated = 18.0")
    print(f"  interior outputs = {interior}; outputs exactly equal to 18.0 = {hits}")
    print(f"  observed output range = [{y.min().item()!r}, {y.max().item()!r}]")


def main():
    print(f"torch {torch.__version__}")
    if hasattr(torch, "musa") and hasattr(torch.musa, "is_available"):
        print(f"torch.musa.is_available() = {torch.musa.is_available()}")

    knobs = _reduced_precision_knobs()
    if knobs:
        for backend_name, backend, knob in knobs:
            print(f"torch.backends.{backend_name}.{knob} = {getattr(backend, knob)}")
    else:
        print("no allow_tf32/allow_hf32 knob under torch.backends.mudnn or .cudnn")

    print("\ntorch defaults:")
    _run_cases()

    print("\nflag_gems kernel:")
    _probe_flag_gems()

    print("\nTF32 truncation probe:")
    _probe_tf32_truncation()

    if not knobs:
        print(
            "\nNo reduced-precision knob was found to turn off; the default "
            "numbers above still show whether the device reference is fp32-exact."
        )
        return

    for _, backend, knob in knobs:
        setattr(backend, knob, False)
    print("\nreduced precision disabled:")
    for backend_name, backend, knob in knobs:
        print(f"  torch.backends.{backend_name}.{knob} = {getattr(backend, knob)}")
    _run_cases()

    print(
        "\nAn fp32-exact reference is ~1e-6; a TF32/SQMMA-like path is ~1e-3.\n"
        "The test's fp32 tolerance is rtol=1.3e-6, atol=1e-4."
    )


if __name__ == "__main__":
    main()
