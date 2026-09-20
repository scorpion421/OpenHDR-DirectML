# OpenHDR-DirectML for MPC-BE (Standalone Mode without SVP)
# Automatic real-time AI SDR-to-HDR upconversion via DirectML
# 100% Portable - dynamically loads bundled packages without hardcoded paths

import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(base_dir, "packages")

for p in (pkg_dir, base_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import vapoursynth as vs
import vs_directml_hdr

core = vs.core
model_path = os.path.join(base_dir, "hdrtvnet_1080p_fp16.onnx")

# Check if running inside MPC-BE VapourSynth Filter
if "VpsFilterSource" in globals():
    clip = VpsFilterSource
    # Convert SDR stream via calibrated DirectML AI engine
    clip = vs_directml_hdr.Convert(
        clip,
        model_path=model_path,
        device_id=0,
    )
    clip.set_output()
