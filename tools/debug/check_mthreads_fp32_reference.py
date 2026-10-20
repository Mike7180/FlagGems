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
        y = conv(x, w, **kwargs)
        ref = conv(x.cpu().double(), w.cpu().double(), **kwargs)
        print(f"  {name:<26} max|device - cpu_fp64| = {_max_err(y, ref):.3e}")


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
    print(
        "  flag_gems.cudnn_convolution groups=2"
        f" max|gems - cpu_fp64| = {_max_err(y, ref):.3e}"
    )


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
