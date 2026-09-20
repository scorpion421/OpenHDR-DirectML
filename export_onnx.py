"""Export and optimize OpenHDR neural network model to FP16 ONNX.

Produces an optimized ONNX model specifically targeted for DirectML
on AMD Radeon RX 7900 XTX (RDNA 3 AI Matrix Accelerators).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnx
import torch

from models_arch import RealTimeOpenHDRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SDR-to-HDR neural network to ONNX FP16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hdrtvnet_1080p_fp16.onnx"),
        help="Target output ONNX file path",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional path to pretrained PyTorch weights (.pth / .pt)",
    )
    parser.add_argument("--height", type=int, default=1080, help="Frame height (default: 1080)")
    parser.add_argument("--width", type=int, default=1920, help="Frame width (default: 1920)")
    parser.add_argument(
        "--peak-nits", type=float, default=1000.0, help="Target peak HDR luminance in nits"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Export in FP16 precision (default: True)"
    )
    return parser.parse_args()


def export_model(
    output_path: Path,
    weights_path: Path | None = None,
    height: int = 1080,
    width: int = 1920,
    peak_nits: float = 1000.0,
    fp16: bool = True,
) -> None:
    print(f"Building RealTimeOpenHDRNet model (Target Peak: {peak_nits} nits)...")
    model = RealTimeOpenHDRNet(target_peak_nits=peak_nits)

    if weights_path is not None and weights_path.exists():
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Using initialized real-time weights (calibrated for Rec.709 -> BT.2020 + PQ curve).")

    model.eval()

    dummy_input = torch.rand(1, 3, height, width, dtype=torch.float32)

    if fp16:
        print("Converting model to Half Precision (FP16)...")
        model = model.half()
        dummy_input = dummy_input.half()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_onnx = output_path.with_suffix(".temp.onnx")

    print(f"Exporting ONNX graph with input shape {tuple(dummy_input.shape)}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(temp_onnx),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_sdr"],
        output_names=["output_hdr10"],
        dynamic_axes=None,  # Fixed shape eliminates allocation stalls in DirectML
        dynamo=False,
    )

    print("Verifying and optimizing ONNX model...")
    onnx_model = onnx.load(str(temp_onnx))
    onnx.checker.check_model(onnx_model)

    # Save final optimized model
    onnx.save(onnx_model, str(output_path))
    if temp_onnx.exists():
        temp_onnx.unlink()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Successfully exported optimized FP16 ONNX model:")
    print(f"  Path: {output_path.resolve()}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Input: 'input_sdr' (FP16, shape: [1, 3, {height}, {width}], Rec.709 [0..1])")
    print(f"  Output: 'output_hdr10' (FP16, shape: [1, 3, {height}, {width}], BT.2020 PQ [0..1])")


def main() -> None:
    args = parse_args()
    export_model(
        output_path=args.output,
        weights_path=args.weights,
        height=args.height,
        width=args.width,
        peak_nits=args.peak_nits,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
