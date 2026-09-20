# OpenHDR-DirectML
**Real-Time AI-Powered SDR-to-HDR Video Upconversion for AMD Radeon (RDNA 3)**

[![DirectML](https://img.shields.io/badge/DirectML-DirectX%2012-blue.svg)](https://github.com/microsoft/DirectML)
[![VapourSynth](https://img.shields.io/badge/VapourSynth-R70%2B-green.svg)](http://www.vapoursynth.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tested on RX 7900 XTX](https://img.shields.io/badge/Tested%20GPU-AMD%20Radeon%20RX%207900%20XTX-red.svg)](https://www.amd.com/en/products/graphics/desktops/radeon-7000-series/amd-radeon-rx-7900xtx.html)

OpenHDR-DirectML is a hardware-accelerated video enhancement pipeline designed to convert standard 8-bit SDR (Rec.709) video streams into high-dynamic-range 10-bit HDR10 (BT.2020 / SMPTE ST 2084 PQ) in real time on AMD Radeon GPUs using **Microsoft DirectML (DirectX 12)** and **VapourSynth**.

It seamlessly integrates into modern media player stacks (**MPC-BE**, **AviSynth Filter / AVSF**, **MPC Video Renderer**) and coexists perfectly with **SVP 4 (SmoothVideo Project)** 120 / 144 FPS motion interpolation.

---

## 1. Pipeline Architecture & Signal Flow

```text
[Source Media: SDR 8-bit Rec.709 @ 24fps]
                        │
                        ▼
      Hardware Video Decoder (MPC Video Decoder)
           (Outputs NV12 / YUV420P8 / P010)
                        │
                        ▼
     VapourSynth Filter (AVSF) / DirectShow Host
                        │
                        ▼
             VapourSynth Core Engine
                        │
  ┌─────────────────────┴──────────────────────────────┐
  │  STAGE 1: OpenHDR-DirectML Filter Node             │
  │  1. Continuous IEC 61966-2-1 Linearization         │
  │  2. DirectML FP16 Tensor Inference on GPU (~5 ms)  │
  │  3. Cinematic S-Curve Tone Mapping (Inky Blacks)   │
  │  4. Adaptive Color Vibrance (+35% Linear Chroma)   │
  │  5. Dynamic Specular Highlight Expansion (Pop)     │
  │  6. Floyd-Steinberg Error-Diffusion Dithering      │
  └─────────────────────┬──────────────────────────────┘
                        │ (Processed HDR stream @ 24fps)
                        ▼
  ┌────────────────────────────────────────────────────┐
  │  STAGE 2: SVP 4 Motion Interpolation               │
  │  - Ingests the 24fps enhanced stream               │
  │  - Calculates motion vectors (core.svp2.SmoothFps) │
  │  - Outputs silky smooth 120 / 144 fps              │
  └─────────────────────┬──────────────────────────────┘
                        │
                        ▼
             MPC Video Renderer (MPCVR)
          (D3D11 Video Processor Hardware Pipeline)
                        │
                        ▼
         DirectX 11 10-bit HDR10 SwapChain
             (R10G10B10A2_UNORM Flip Discard)
                        │
                        ▼
               HDR Display / Monitor
        (e.g., BenQ MOBIUZ EX3415R DisplayHDR 400)
```

---

## 2. Core Technological Highlights

### Continuous IEC 61966-2-1 Linearization (Zero Black Tearing)
Standard neural tone mappers often apply brutal black floor thresholds that clip dark values below 1%, causing severe solarization and jagged black blotches across dark clothing. OpenHDR-DirectML employs a smooth continuous linear toe function near zero, ensuring natural shadow gradation with zero black tearing.

### Cinematic S-Curve Contrast Engine
Prevents washed-out, milky gray veils by anchoring deep inky blacks firmly while smoothly lifting midtones for crisp daylight clarity.

### Dynamic Specular Highlight Expansion
Analyzes specular elements (lamps, chrome reflections, direct sunlight, explosions) and rolls them smoothly onto the display's physical peak brightness (e.g. 400 Nits on DisplayHDR 400 panels).

### Adaptive Color Vibrance (+35%)
Dynamically expands chromaticity in linear light, providing richer, deeper saturation to muted background colors while carefully protecting natural skin tones from over-saturation or tint shift.

### Native 24 fps Inference Architecture
Because the OpenHDR node processes video **before** SVP motion interpolation, the neural network computes only **24 frames per second**, keeping GPU load at just ~8% to 15% and leaving ample compute resources for SVP 4 high-refresh (120/144 fps) interpolation and MPC Video Renderer shaders.

---

## 3. Hardware Benchmarking on AMD Radeon RX 7900 XTX

Benchmarked on **AMD Radeon RX 7900 XTX** (Navi 31, RDNA 3 AI Matrix Accelerators) via Microsoft DirectML FP16 execution provider:

| Video Resolution | Mean Latency | Median Latency | Throughput | GPU Load @ 24 fps |
| :--- | :--- | :--- | :--- | :--- |
| **1080p (1920x1080)** | **5.40 ms** | **5.37 ms** | **185.2 FPS** | **~13.0%** |
| **Ultrawide (1920x800)** | **7.97 ms** | **7.82 ms** | **125.5 FPS** | **~19.1%** |

---

## 4. Quickstart Guide

### Option A: Plug & Play Bundle (Recommended)
1. Download the pre-built `OpenHDR-for-MPCBE.zip` from the [Releases](https://github.com/scorpion421/OpenHDR-DirectML/releases) tab.
2. Extract the archive into your MPC-BE folder (where `mpc-be64.exe` is located).
3. If using **SVP 4**, open `%APPDATA%\SVP4\override.js` and add:
   ```javascript
   global.baseScript = global.baseScript.replace(
       "def interpolate(clip):",
       "import sys; sys.path.insert(0, 'D:/Apps/MPCBE/OpenHDR/packages'); sys.path.insert(0, 'D:/Apps/MPCBE/OpenHDR')\n" +
       "import vs_directml_hdr, importlib; importlib.reload(vs_directml_hdr)\n" +
       "def interpolate(clip):\n" +
       "    clip = vs_directml_hdr.Convert(clip, model_path='D:/Apps/MPCBE/OpenHDR/hdrtvnet_1080p_fp16.onnx')\n"
   );
   ```
   *(Update `D:/Apps/MPCBE/OpenHDR` to your path).*
4. Start playback in MPC-BE.

### Option B: Build & Export from Source
1. Clone this repository:
   ```powershell
   git clone https://github.com/scorpion421/OpenHDR-DirectML.git
   cd OpenHDR-DirectML
   ```
2. Set up virtual environment and install requirements:
   ```powershell
   .\setup_env.ps1
   ```
3. Export and compile the DirectML ONNX model:
   ```powershell
   python build_cinematic_hdr.py
   ```
4. Run the automated pipeline benchmark:
   ```powershell
   python benchmark_dml.py
   ```

---

## 5. Verification & Telemetry

1. **OSD Telemetry**: During playback in MPC-BE, press `Ctrl + J` to inspect the MPC Video Renderer OSD:
   - `Color: RGB 10-bit HDR10: On`
   - `VideoProcessor: D3D11 VP, output to R10G10B10A2_UNORM`
   - `Presentation: Flip discard, R10G10B10A2_UNORM`
2. **GPU Activity**: Check Windows Task Manager (`Ctrl + Shift + Esc`) -> GPU -> Compute:
   - Verify steady 8% to 15% compute activity during playback, dropping to 0% when paused.
3. **Live A/B Testing**: In `%APPDATA%\SVP4\override.js`, comment out the `clip = vs_directml_hdr.Convert(...)` line with `//` and save:
   - SVP reloads live during playback without restarting playback. Notice how contrast flattens and highlights dim. Removing `//` immediately restores punchy HDR depth.

---

## 6. Project Structure

```text
OpenHDR-DirectML/
├── models_arch.py             # PyTorch SDR-to-HDR neural architecture
├── build_cinematic_hdr.py     # Calibrated S-curve ONNX export tool
├── hdrtvnet_1080p_fp16.onnx   # Production DirectML FP16 model
├── vs_directml_hdr.py         # Core VapourSynth filter node implementation
├── benchmark_dml.py           # Hardware latency benchmark tool
├── test_pipeline.py           # Verification and pattern generator suite
├── requirements.txt           # Python package dependencies
├── setup_env.ps1              # Automated local venv environment builder
├── install.ps1                # Automated benchmark and setup runner
└── README.md                  # Master documentation
```

---

## 7. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
