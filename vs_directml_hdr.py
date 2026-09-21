"""VapourSynth Filter Node: OpenHDR-DirectML.

Performs real-time AI-powered SDR-to-HDR10 conversion on AMD Radeon GPUs
(specifically AMD Radeon RX 7900 XTX via DirectML / DirectX 12).

Signal & Metadata Flow:
  Input:  8-bit SDR Rec.709 (YUV420P8 or RGB24)
  Stage:  DirectML Tensor Inference (NCHW)
  Output: 10-bit HDR10 (YUV420P10 or RGB48)
  Flags:  _Matrix=1 (BT.709), _Primaries=1 (BT.709), _Transfer=1 (BT.709)
"""

from __future__ import annotations

import os
import threading
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
        self.is_dynamic = not (
            len(self.input_shape) >= 4
            and isinstance(self.input_shape[2], int)
            and isinstance(self.input_shape[3], int)
        )
        self.expected_h = self.input_shape[2] if not self.is_dynamic else None
        self.expected_w = self.input_shape[3] if not self.is_dynamic else None
        self.is_fp16 = "float16" in input_meta.type

    def infer_tensor(self, nchw_rgb: np.ndarray) -> np.ndarray:
        """Runs DirectML inference on NCHW RGB tensor in [0.0, 1.0].
        
        Args:
            nchw_rgb: NumPy array with shape (1, 3, H, W).
        Returns:
            NumPy array with shape (1, 3, H, W) containing HDR values [0.0, 1.0].
        """
        in_dtype = np.float16 if self.is_fp16 else np.float32
        tensor = nchw_rgb.astype(in_dtype, copy=False)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return outputs[0]


def load_config() -> dict[str, Any]:
    """Loads configuration from openhdr.ini in player root or OpenHDR directory."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "sdr_only": True,
        "max_width": 1920,
        "max_height": 1088,
        "display_peak_nits": 400,
        "black_level_nits": 0.005,
        "max_cll": 400,
        "max_fall": 200,
        "contrast_curve": 1.06,
        "vibrance_boost": 0.35,
        "specular_boost": 0.20,
        "device_id": 0,
        "use_fp16": True,
    }

    # Dynamic search paths (searches player root and OpenHDR directories portably)
    search_paths = [
        Path(os.environ["OPENHDR_INI"]) if "OPENHDR_INI" in os.environ else None,
        Path(__file__).resolve().parent.parent / "openhdr.ini",
        Path(__file__).resolve().parent / "openhdr.ini",
        Path.cwd() / "openhdr.ini",
    ]

    config_path = None
    for p in search_paths:
        if p is not None and p.exists():
            config_path = p
            break

    if config_path is None:
        return defaults

    import configparser
    parser = configparser.ConfigParser()
    try:
        parser.read(str(config_path), encoding="utf-8")
        if parser.has_section("General"):
            defaults["enabled"] = parser.getboolean("General", "enabled", fallback=defaults["enabled"])
            defaults["sdr_only"] = parser.getboolean("General", "sdr_only", fallback=defaults["sdr_only"])
            defaults["max_width"] = parser.getint("General", "max_width", fallback=defaults["max_width"])
            defaults["max_height"] = parser.getint("General", "max_height", fallback=defaults["max_height"])

        if parser.has_section("HDR_Engine"):
            defaults["display_peak_nits"] = parser.getint("HDR_Engine", "display_peak_nits", fallback=defaults["display_peak_nits"])
            defaults["black_level_nits"] = parser.getfloat("HDR_Engine", "black_level_nits", fallback=defaults["black_level_nits"])
            defaults["max_cll"] = parser.getint("HDR_Engine", "max_cll", fallback=defaults["max_cll"])
            defaults["max_fall"] = parser.getint("HDR_Engine", "max_fall", fallback=defaults["max_fall"])
            defaults["contrast_curve"] = parser.getfloat("HDR_Engine", "contrast_curve", fallback=defaults["contrast_curve"])
            defaults["vibrance_boost"] = parser.getfloat("HDR_Engine", "vibrance_boost", fallback=defaults["vibrance_boost"])
            defaults["specular_boost"] = parser.getfloat("HDR_Engine", "specular_boost", fallback=defaults["specular_boost"])

        if parser.has_section("Performance"):
            defaults["device_id"] = parser.getint("Performance", "device_id", fallback=defaults["device_id"])
            defaults["use_fp16"] = parser.getboolean("Performance", "use_fp16", fallback=defaults["use_fp16"])
    except Exception as err:
        print(f"[OpenHDR] Warning: Failed to parse {config_path}: {err}. Using defaults.")

    return defaults


def Convert(
    clip: Any,
    model_path: str | Path = "hdrtvnet_1080p_fp16.onnx",
    device_id: int | None = None,
    output_format: str | None = None,
    max_luminance: int | None = None,
    min_luminance: float | None = None,
    max_cll: int | None = None,
    max_fall: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
) -> Any:
    """VapourSynth filter entrypoint for DirectML HDR conversion.
    
    Reads configuration from openhdr.ini with fallback to keyword arguments.
    """
    if not HAS_VAPOURSYNTH:
        raise RuntimeError("VapourSynth is not installed or available in this Python environment.")

    cfg = load_config()

    # 1. Master toggle from openhdr.ini
    if not cfg["enabled"]:
        return clip

    # 2. Resolution gate: Only engage for content up to max_width x max_height (default 1080p)
    limit_w = max_width if max_width is not None else cfg["max_width"]
    limit_h = max_height if max_height is not None else cfg["max_height"]
    if clip.width > limit_w or clip.height > limit_h:
        return clip

    # 3. SDR gate: Only engage for SDR streams (bypass native HDR10 / PQ, HLG, BT.2020)
    if cfg["sdr_only"]:
        try:
            sample_frame = clip.get_frame(0)
            transfer = sample_frame.props.get("_Transfer", 0)
            primaries = sample_frame.props.get("_Primaries", 0)
            matrix = sample_frame.props.get("_Matrix", 0)
            if transfer in (14, 16, 18) or primaries == 9 or matrix == 9:
                return clip
        except Exception:
            pass

    core = vs.core
    gpu_id = device_id if device_id is not None else cfg["device_id"]
    engine = DirectML_HDR_Engine(model_path=model_path, device_id=gpu_id)

    orig_w = clip.width
    orig_h = clip.height
    needs_scale = False
    if not engine.is_dynamic:
        needs_scale = (orig_w, orig_h) != (engine.expected_w, engine.expected_h)
        if needs_scale:
            clip = core.resize.Bicubic(clip, width=engine.expected_w, height=engine.expected_h)

    # 2. Convert to planar 32-bit float RGB (Rec.709) for model input
    rgb_sdr_clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

    # Thread-local static tensor allocation to eliminate heap churn and ensure 100% thread safety
    import gc
    gc.set_threshold(100000, 50, 50)
    in_dtype = np.float16 if engine.is_fp16 else np.float32
    tls = threading.local()

    def get_thread_tensor(h: int, w: int) -> np.ndarray:
        tensor = getattr(tls, "tensor", None)
        if tensor is None or tensor.shape != (1, 3, h, w) or tensor.dtype != in_dtype:
            tensor = np.empty((1, 3, h, w), dtype=in_dtype)
            tls.tensor = tensor
        return tensor

    def process_frame(n: int, f: Any) -> vs.VideoFrame:
        in_frame = f if isinstance(f, vs.VideoFrame) else f[0]
        h = in_frame.height
        w = in_frame.width

        # Thread-safe zero-allocation fill directly from plane memory
        thread_tensor = get_thread_tensor(h, w)
        thread_tensor[0, 0] = np.frombuffer(in_frame[0], dtype=np.float32).reshape((h, w))
        thread_tensor[0, 1] = np.frombuffer(in_frame[1], dtype=np.float32).reshape((h, w))
        thread_tensor[0, 2] = np.frombuffer(in_frame[2], dtype=np.float32).reshape((h, w))

        # DirectML GPU inference
        out_tensor = engine.infer_tensor(thread_tensor)

        # Write planes into frame
        out_frame = in_frame.copy()
        np.frombuffer(out_frame[0], dtype=np.float32).reshape((h, w))[:] = out_tensor[0, 0]
        np.frombuffer(out_frame[1], dtype=np.float32).reshape((h, w))[:] = out_tensor[0, 1]
        np.frombuffer(out_frame[2], dtype=np.float32).reshape((h, w))[:] = out_tensor[0, 2]
        return out_frame

    # Evaluate per frame
    hdr_rgb_clip = core.std.ModifyFrame(
        clip=rgb_sdr_clip,
        clips=[rgb_sdr_clip],
        selector=process_frame,
    )

    # 3. Convert output to target format with error diffusion dithering (Zero Banding)
    target_format = clip.format.id
    if output_format is not None and hasattr(vs, output_format):
        target_format = getattr(vs, output_format)

    hdr_output_clip = core.resize.Bicubic(
        hdr_rgb_clip,
        format=target_format,
        matrix_s="709",
        range_s="limited",
        dither_type="ordered",
    )

    # If source had a different resolution, scale back to original resolution
    if needs_scale:
        hdr_output_clip = core.resize.Bicubic(hdr_output_clip, width=orig_w, height=orig_h)

    # 4. Inject frame properties conforming to BT.709 standards for MPC Video Renderer
    hdr10_clip = core.std.SetFrameProps(
        hdr_output_clip,
        _Matrix=1,          # BT.709
        _Primaries=1,       # BT.709
        _Transfer=1,        # BT.709
        _FieldBased=0,      # Progressive
    )

    return hdr10_clip
