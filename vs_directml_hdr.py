"""VapourSynth Filter Node: OpenHDR-DirectML.

Performs real-time AI-powered SDR-to-HDR10 conversion on AMD Radeon GPUs
(specifically AMD Radeon RX 7900 XTX via DirectML / DirectX 12).

Signal & Metadata Flow:
  Input:  8-bit SDR Rec.709 (YUV420P8 or RGB24)
  Stage:  DirectML FP16 Tensor Inference (NCHW)
  Output: 10-bit HDR10 (YUV420P10 or RGB48)
  Flags:  _Matrix=9 (BT.2020), _Primaries=9 (BT.2020), _Transfer=16 (PQ)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:
    import vapoursynth as vs
    HAS_VAPOURSYNTH = True
except ImportError:
    vs = None
    HAS_VAPOURSYNTH = False


class DirectML_HDR_Engine:
    """Inference session manager for DirectML SDR-to-HDR models."""

    def __init__(self, model_path: str | Path, device_id: int = 0) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"HDR ONNX model not found: {self.model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ort.get_available_providers()
        selected_providers = []
        provider_options = []

        if "DmlExecutionProvider" in providers:
            selected_providers.append("DmlExecutionProvider")
            provider_options.append({"device_id": device_id})
        else:
            print("WARNING: DmlExecutionProvider not available, falling back to CPU.")
            selected_providers.append("CPUExecutionProvider")
            provider_options.append({})

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=selected_providers,
            provider_options=provider_options,
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = input_meta.shape  # Expected [1, 3, H, W]
        self.expected_h = self.input_shape[2] if len(self.input_shape) >= 4 else 1080
        self.expected_w = self.input_shape[3] if len(self.input_shape) >= 4 else 1920
        self.is_fp16 = "float16" in input_meta.type

    def infer_tensor(self, nchw_rgb: np.ndarray) -> np.ndarray:
        """Runs DirectML inference on NCHW RGB tensor in [0.0, 1.0].
        
        Args:
            nchw_rgb: NumPy array with shape (1, 3, H, W).
        Returns:
            NumPy array with shape (1, 3, H, W) containing BT.2020 PQ values [0.0, 1.0].
        """
        in_dtype = np.float16 if self.is_fp16 else np.float32
        tensor = nchw_rgb.astype(in_dtype, copy=False)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return outputs[0]


def Convert(
    clip: Any,
    model_path: str | Path = "hdrtvnet_1080p_fp16.onnx",
    device_id: int = 0,
    output_format: str = "YUV420P10",
    max_luminance: int = 1000,
    min_luminance: float = 0.005,
    max_cll: int = 1000,
    max_fall: int = 400,
) -> Any:
    """VapourSynth filter entrypoint for DirectML HDR conversion.
    
    Args:
        clip: Input VapourSynth VideoNode (8-bit SDR Rec.709).
        model_path: Path to optimized FP16 ONNX model.
        device_id: DirectML GPU device ID (0 = discrete GPU).
        output_format: Output pixel format ('YUV420P10' or 'RGB48').
        max_luminance: Mastering display peak luminance in nits.
        min_luminance: Mastering display black level in nits.
        max_cll: Maximum Content Light Level in nits.
        max_fall: Maximum Frame-Average Light Level in nits.
    Returns:
        Converted 10-bit HDR10 VideoNode with full HDR10 frame properties.
    """
    if not HAS_VAPOURSYNTH:
        raise RuntimeError("VapourSynth is not installed or available in this Python environment.")

    core = vs.core
    engine = DirectML_HDR_Engine(model_path=model_path, device_id=device_id)

    # 1. Standardize input clip resolution to match model tensor shape
    in_w = clip.width
    in_h = clip.height
    if (in_w, in_h) != (engine.expected_w, engine.expected_h):
        clip = core.resize.Bicubic(clip, width=engine.expected_w, height=engine.expected_h)

    # 2. Convert to planar 32-bit float RGB (Rec.709) for model input
    rgb_sdr_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

    def process_frame(n: int, f: Any) -> vs.VideoFrame:
        in_frame = f if isinstance(f, vs.VideoFrame) else f[0]
        h = in_frame.height
        w = in_frame.width

        # Read R, G, B planes via memoryview with zero-copy
        r_plane = np.frombuffer(in_frame[0], dtype=np.float32).reshape((h, w))
        g_plane = np.frombuffer(in_frame[1], dtype=np.float32).reshape((h, w))
        b_plane = np.frombuffer(in_frame[2], dtype=np.float32).reshape((h, w))

        # Stack into [1, 3, H, W] tensor
        rgb_tensor = np.stack([r_plane, g_plane, b_plane], axis=0)[np.newaxis, ...]

        # DirectML GPU inference
        out_tensor = engine.infer_tensor(rgb_tensor)

        # Write planes into copy of frame
        out_frame = in_frame.copy()
        out_r = out_tensor[0, 0].astype(np.float32, copy=False)
        out_g = out_tensor[0, 1].astype(np.float32, copy=False)
        out_b = out_tensor[0, 2].astype(np.float32, copy=False)

        np.frombuffer(out_frame[0], dtype=np.float32).reshape((h, w))[:] = out_r
        np.frombuffer(out_frame[1], dtype=np.float32).reshape((h, w))[:] = out_g
        np.frombuffer(out_frame[2], dtype=np.float32).reshape((h, w))[:] = out_b
        return out_frame

    # Evaluate per frame
    hdr_rgb_clip = core.std.ModifyFrame(
        clip=rgb_sdr_clip,
        clips=[rgb_sdr_clip],
        selector=process_frame,
    )

    # 3. Convert output from BT.2020 RGB to 10-bit YUV420P10 (or RGB48)
    if output_format.upper() == "RGB48":
        hdr_output_clip = core.resize.Bicubic(hdr_rgb_clip, format=vs.RGB48)
    else:
        # Standard YUV420P10 with BT.2020 non-constant luminance matrix and limited range
        hdr_output_clip = core.resize.Bicubic(
            hdr_rgb_clip,
            format=vs.YUV420P10,
            matrix_s="2020ncl",
            range_s="limited",
        )

    # 4. Inject HDR10 frame properties conforming to SMPTE ST 2084 / BT.2020 standards
    hdr10_clip = core.std.SetFrameProps(
        hdr_output_clip,
        _Matrix=9,          # BT.2020 non-constant luminance
        _Primaries=9,       # BT.2020
        _Transfer=16,       # SMPTE ST 2084 / PQ
        _FieldBased=0,      # Progressive
        MasteringDisplayMinLuminance=int(min_luminance * 10000),  # 0.0001 nit units
        MasteringDisplayMaxLuminance=int(max_luminance * 10000),
        MaxCLL=max_cll,
        MaxFALL=max_fall,
    )

    return hdr10_clip
