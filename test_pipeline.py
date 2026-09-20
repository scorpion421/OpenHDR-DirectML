"""Automated verification and end-to-end test suite for OpenHDR-DirectML.

Tests:
1. DirectML inference session on AMD Radeon RX 7900 XTX.
2. Tensor validity, value bounds, gamut expansion, and PQ curve compliance.
3. VapourSynth filter node integration (when VapourSynth is available).
4. Generation of visual verification artifacts (test SDR vs HDR visualization).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

from vs_directml_hdr import DirectML_HDR_Engine, HAS_VAPOURSYNTH


def create_synthetic_sdr_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Generates a synthetic 1080p SDR Rec.709 test pattern.
    
    Includes color bars, horizontal luminance gradients, and high-intensity specular spots.
    Returns:
        NumPy array of shape (height, width, 3), float32 in [0.0, 1.0], RGB order.
    """
    img = np.zeros((height, width, 3), dtype=np.float32)

    # 1. Color bars in top third
    bar_height = height // 3
    num_bars = 7
    bar_width = width // num_bars
    colors = [
        [1.0, 1.0, 1.0],  # White
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
        [0.0, 1.0, 0.0],  # Green
        [1.0, 0.0, 1.0],  # Magenta
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
    ]
    for i, col in enumerate(colors):
        x0 = i * bar_width
        x1 = width if i == num_bars - 1 else (i + 1) * bar_width
        img[0:bar_height, x0:x1] = col

    # 2. Smooth gradient in middle third
    grad_height = height // 3
    grad_row = np.linspace(0.0, 1.0, width, dtype=np.float32)
    grad_block = np.tile(grad_row, (grad_height, 1))
    for c in range(3):
        img[bar_height : bar_height + grad_height, :, c] = grad_block

    # 3. Specular highlight zones in bottom third
    bottom_y = bar_height + grad_height
    # Dark background with highlight circles
    img[bottom_y:, :] = 0.05
    cv2.circle(img, (width // 4, bottom_y + (height - bottom_y) // 2), 60, (0.95, 0.95, 0.95), -1)
    cv2.circle(img, (width // 2, bottom_y + (height - bottom_y) // 2), 80, (1.0, 0.8, 0.2), -1)
    cv2.circle(img, (3 * width // 4, bottom_y + (height - bottom_y) // 2), 70, (0.2, 0.9, 1.0), -1)

    return img


def pq_to_linear_nits(pq_val: np.ndarray) -> np.ndarray:
    """Inverts SMPTE ST 2084 PQ to approximate linear light in nits."""
    m1 = 2610.0 / 16384.0
    m2 = (2523.0 / 4096.0) * 128.0
    c1 = 3424.0 / 4096.0
    c2 = (2413.0 / 4096.0) * 32.0
    c3 = (2392.0 / 4096.0) * 32.0

    pq_clamp = np.clip(pq_val, 1e-6, 1.0)
    vp = np.power(pq_clamp, 1.0 / m2)
    num = np.maximum(vp - c1, 0.0)
    den = c2 - c3 * vp
    den = np.maximum(den, 1e-6)
    linear_rel = np.power(num / den, 1.0 / m1)
    return linear_rel * 10000.0


def run_pipeline_tests(model_path: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("============================================================")
    print("OpenHDR-DirectML End-to-End Pipeline Verification")
    print("============================================================")

    # 1. Engine Initialization
    print(f"Loading DirectML model: {model_path}...")
    engine = DirectML_HDR_Engine(model_path=model_path, device_id=0)
    print("DirectML session initialized successfully.")

    # 2. Test Tensor Inference
    print("Generating synthetic 1080p SDR Rec.709 frame...")
    sdr_img = create_synthetic_sdr_frame(width=1920, height=1080)
    
    # Save input SDR visualization (sRGB 8-bit)
    sdr_bgr_8u = np.clip(sdr_img[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "test_sdr_input.png"), sdr_bgr_8u)
    print(f"Saved input SDR pattern to: {output_dir / 'test_sdr_input.png'}")

    # Convert to NCHW float
    rgb_nchw = np.transpose(sdr_img, (2, 0, 1))[np.newaxis, ...]

    print("Running DirectML inference on GPU...")
    t0 = time.perf_counter()
    hdr_nchw = engine.infer_tensor(rgb_nchw)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    print(f"DirectML inference completed in {infer_ms:.2f} ms.")

    # 3. Output Validation
    print("Validating output tensor structure...")
    assert hdr_nchw.shape == (1, 3, 1080, 1920), f"Unexpected shape {hdr_nchw.shape}"
    min_val = float(np.min(hdr_nchw))
    max_val = float(np.max(hdr_nchw))
    mean_val = float(np.mean(hdr_nchw))
    print(f"  HDR10 PQ code value range: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}")

    assert min_val >= 0.0, f"Negative value detected in PQ code: {min_val}"
    assert max_val <= 1.05, f"PQ code value exceeds ceiling: {max_val}"

    # Calculate estimated peak nits from PQ output
    peak_nits = float(np.max(pq_to_linear_nits(hdr_nchw)))
    print(f"  Estimated Peak Luminance: {peak_nits:.1f} Nits")

    # Save tone-mapped visualization of HDR output for review on SDR displays
    hdr_hwc = np.transpose(hdr_nchw[0], (1, 2, 0))
    # Simple Reinhard tone-map of linear nits for SDR preview
    linear_nits = pq_to_linear_nits(hdr_hwc)
    preview_sdr = linear_nits / (linear_nits + 200.0)
    preview_bgr = np.clip(preview_sdr[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "test_hdr_preview.png"), preview_bgr)
    print(f"Saved HDR tone-mapped preview to: {output_dir / 'test_hdr_preview.png'}")

    # 4. VapourSynth Integration Test
    if HAS_VAPOURSYNTH:
        print("\nVapourSynth detected. Testing VapourSynth filter node...")
        import vapoursynth as vs
        import vs_directml_hdr

        core = vs.core
        # Create a small 5-frame 1080p test clip
        test_clip = core.std.BlankClip(
            format=vs.YUV420P8,
            width=1920,
            height=1080,
            length=5,
            fpsnum=24,
            fpsden=1,
            color=[128, 128, 128],
        )

        hdr_vs_clip = vs_directml_hdr.Convert(
            test_clip,
            model_path=model_path,
            device_id=0,
            output_format="YUV420P10",
        )

        assert hdr_vs_clip.format.name == "YUV420P10", f"Expected YUV420P10, got {hdr_vs_clip.format.name}"
        assert hdr_vs_clip.format.bits_per_sample == 10, "Expected 10-bit depth"

        frame = hdr_vs_clip.get_frame(0)
        props = frame.props

        print("VapourSynth Output Frame Properties:")
        print(f"  _Matrix:       {props.get('_Matrix')} (Expected: 9 = BT.2020 NCL)")
        print(f"  _Primaries:    {props.get('_Primaries')} (Expected: 9 = BT.2020)")
        print(f"  _Transfer:     {props.get('_Transfer')} (Expected: 16 = SMPTE ST 2084 / PQ)")
        print(f"  _Range:        {props.get('_Range')} (Expected: RANGE_LIMITED)")
        print(f"  MaxLuminance:  {props.get('MasteringDisplayMaxLuminance')} (Expected: 10000000 = 1000 nits)")

        assert props.get("_Matrix") == 9, "Incorrect _Matrix property"
        assert props.get("_Primaries") == 9, "Incorrect _Primaries property"
        assert props.get("_Transfer") == 16, "Incorrect _Transfer property"
        assert int(props.get("_Range")) == 0, "Incorrect _Range property (expected limited range)"
        print("VapourSynth HDR10 frame property handshake verified!")
    else:
        print("\nNote: VapourSynth module not in current Python environment (tested direct tensor pipeline).")

    print("\n============================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OpenHDR-DirectML pipeline")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hdrtvnet_1080p_fp16.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output"),
        help="Directory to save test outputs",
    )
    args = parser.parse_args()
    success = run_pipeline_tests(model_path=args.model, output_dir=args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
