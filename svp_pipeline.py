"""SVP 4 (SmoothVideo Project) & AviSynth Filter (AVSF) Integration Script.

This script demonstrates how to integrate OpenHDR-DirectML into the
SVP 4 / MPC-BE VapourSynth pipeline.

Placement:
  The AI HDR node is inserted BEFORE core.svp2.SmoothFps.
  This allows the DirectML neural network to process frames at the native
  source framerate (e.g. 24 fps / 41.67 ms frame budget), while SVP
  interpolates the resulting 10-bit HDR10 frames up to 120/144 fps.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure local script directory is in Python module search path
project_dir = str(Path(__file__).resolve().parent)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import vapoursynth as vs
import vs_directml_hdr

core = vs.core

def process_stream(
    source_clip: vs.VideoNode,
    model_name: str = "hdrtvnet_1080p_fp16.onnx",
    device_id: int = 0,
    enable_svp: bool = True,
    svp_profile: dict | None = None,
) -> vs.VideoNode:
    """Processes incoming SDR video stream from AVSF/DirectShow through OpenHDR + SVP.
    
    Args:
        source_clip: Input SDR video node from AVSF (e.g. VpsFilterSource).
        model_name: Model filename located in project directory.
        device_id: DirectML GPU device (0 = discrete AMD Radeon RX 7900 XTX).
        enable_svp: Whether to apply SVP motion interpolation after HDR conversion.
        svp_profile: Optional dictionary containing custom SVP parameters.
    Returns:
        High-framerate 10-bit HDR10 video node.
    """
    model_path = Path(project_dir) / model_name

    # STAGE 1: Real-time SDR-to-HDR10 AI Conversion at native framerate (24 fps)
    hdr_clip = vs_directml_hdr.Convert(
        source_clip,
        model_path=model_path,
        device_id=device_id,
        output_format="YUV420P10",
        max_luminance=1000,
        min_luminance=0.005,
        max_cll=1000,
        max_fall=400,
    )

    if not enable_svp or not hasattr(core, "svp1") or not hasattr(core, "svp2"):
        # If SVP plugins are not loaded or disabled, output 24 fps HDR10 directly
        return hdr_clip

    # STAGE 2: SVP 4 Motion Interpolation on the 10-bit HDR10 stream
    # Prepare 8-bit motion analysis clip (SVP standard recommendation for performance)
    analysis_clip = core.resize.Bicubic(hdr_clip, format=vs.YUV420P8)

    super_params = "{scale:{up:0},gpu:1,rc:false}"
    analyse_params = "{block:{w:32,h:32},main:{search:{type:4,distance:-4,sort:true}},special:{penalty:{lambda:2.0}}}"
    smoothfps_params = "{rate:{num:5,den:1},algo:13,mask:{cover:100,area:0},scene:{mode:0}}"

    if svp_profile:
        super_params = svp_profile.get("super", super_params)
        analyse_params = svp_profile.get("analyse", analyse_params)
        smoothfps_params = svp_profile.get("smoothfps", smoothfps_params)

    super_data = core.svp1.Super(analysis_clip, super_params)
    vectors = core.svp1.Analyse(super_data["clip"], super_data["data"], analysis_clip, analyse_params)

    smooth_hdr = core.svp2.SmoothFps(
        hdr_clip,
        super_data["clip"],
        super_data["data"],
        vectors["clip"],
        vectors["data"],
        smoothfps_params,
        src=hdr_clip,
    )

    # Preserve essential HDR10 properties through SVP filter graph
    smooth_hdr = core.std.SetFrameProps(
        smooth_hdr,
        _Matrix=9,
        _Primaries=9,
        _Transfer=16,
        _ColorRange=1,
    )

    return smooth_hdr


# Standard entry point when invoked directly by AviSynth Filter / SVP 4
if "VpsFilterSource" in globals():
    clip = VpsFilterSource
    clip = process_stream(clip)
    clip.set_output()
