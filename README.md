# OpenHDR-DirectML
**Real-Time AI-Powered SDR-to-HDR Video Upconversion for AMD Radeon (RDNA 3)**

[![DirectML](https://img.shields.io/badge/DirectML-DirectX%2012-blue.svg)](https://github.com/microsoft/DirectML)
[![VapourSynth](https://img.shields.io/badge/VapourSynth-R70%2B-green.svg)](http://www.vapoursynth.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tested on RX 7900 XTX](https://img.shields.io/badge/Tested%20GPU-AMD%20Radeon%20RX%207900%20XTX-red.svg)](https://www.amd.com/en/products/graphics/desktops/radeon-7000-series/amd-radeon-rx-7900xtx.html)

OpenHDR-DirectML is a hardware-accelerated video enhancement pipeline designed to convert standard 8-bit SDR (Rec.709) video streams into high-dynamic-range 10-bit HDR10 (BT.2020 / SMPTE ST 2084 PQ) in real time on AMD Radeon GPUs using **Microsoft DirectML (DirectX 12)** and **VapourSynth**.

Supports two flexible deployment options:
- **Standalone Mode (No SVP Required)**: Real-time AI HDR upconversion at native video framerate (24 / 30 / 60 FPS) directly inside MPC-BE.
- **SVP 4 Integration Mode**: Real-time AI HDR upconversion combined with 120 / 144 FPS high-refresh motion interpolation.

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
                        │
         ┌──────────────┴──────────────┐
         │ (Standalone Mode)           │ (SVP 4 Coexistence)
         │ Outputs 24fps HDR           │ Interpolates to 120/144fps
         ▼                             ▼
  [Direct Delivery]             [SVP 4 SmoothFps Engine]
         │                             │
         └──────────────┬──────────────┘
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

### Option 1: Standalone MPC-BE (No SVP Required)
1. Download `OpenHDR-for-MPCBE.zip` from the [Releases](https://github.com/scorpion421/OpenHDR-DirectML/releases) tab.
2. Extract all files into your MPC-BE directory (where `mpc-be64.exe` is located):
   - `OpenHDR\` (folder)
   - `vapoursynth_filter_64.ax`
   - `vapoursynth_filter.ini`
3. In MPC-BE, go to **View** -> **Options** -> **External Filters** -> **Add Filter...** -> Select `vapoursynth_filter_64.ax` -> set to **Prefer**.
4. In **Video**, ensure **MPC Video Renderer (MPCVR)** is selected.
5. Start playback.

### Option 2: SVP 4 Coexistence Mode (AI HDR + 120/144 FPS)
1. Copy only the `OpenHDR\` folder from `OpenHDR-for-MPCBE.zip` into your MPC-BE folder.
2. Open `%APPDATA%\SVP4\override.js` and insert inside `override = function()`:
   ```javascript
   global.baseScript = global.baseScript.replace(
       "def interpolate(clip):",
       "import sys; sys.path.insert(0, 'D:/Apps/MPCBE/OpenHDR/packages'); sys.path.insert(0, 'D:/Apps/MPCBE/OpenHDR')\n" +
       "import vs_directml_hdr, importlib; importlib.reload(vs_directml_hdr)\n" +
       "def interpolate(clip):\n" +
       "    clip = vs_directml_hdr.Convert(clip, model_path='D:/Apps/MPCBE/OpenHDR/hdrtvnet_1080p_fp16.onnx')\n"
   );
   ```
   *(Update `D:/Apps/MPCBE/OpenHDR` to your actual path).*
3. Play video. SVP 4 executes AI conversion at 24 FPS and interpolates smoothly to 120/144 FPS in 10-bit HDR10.

---

## 5. Configuration (`openhdr.ini`)

OpenHDR supports hot-reload configuration via `openhdr.ini` placed in the player directory (e.g. next to `mpc-be64.exe`) or in `OpenHDR/openhdr.ini`:

```ini
[General]
enabled = true              ; Master toggle for OpenHDR DirectML conversion
sdr_only = true             ; Only process SDR video (bypasses native HDR10/HLG)
max_width = 1920            ; Resolution gate limit
max_height = 1088

[HDR_Engine]
display_peak_nits = 400     ; Peak mastering display luminance in nits
black_level_nits = 0.005    ; Target display black level
max_cll = 400               ; Content light level metadata
max_fall = 200
contrast_curve = 1.06       ; Rich inky blacks (1.00 = standard linear)
vibrance_boost = 0.35       ; Midtone color vibrance boost (shadow-protected)
specular_boost = 0.20       ; Highlight pop intensity

[Performance]
device_id = 0               ; GPU adapter (0 = discrete GPU, e.g. RX 7900 XTX)
use_fp16 = true             ; FP16 tensor acceleration
```

### Hardware Upscaling & Super Resolution (`VPSuperResolution`):
Resolution upscaling to fullscreen display resolutions (1440p, 4K UHD, Ultrawide) is handled natively in hardware by the AMD Radeon GPU and Direct3D 11 Video Processor (D3D11 VP) in MPC Video Renderer. This eliminates DirectShow memory copy latency and delivers rock-solid 144+ FPS presentation without skipped frames.

#### How to Enable / Disable MPC-VR Hardware Super Resolution:
* **Option A — MPC-BE GUI**: During video playback, right-click video &rarr; **Filters** &rarr; **MPC Video Renderer** &rarr; Check or uncheck **Super Resolution**.
* **Option B — Windows Registry (PowerShell)**:
  * Enable (Value `1`):
    ```powershell
    Set-ItemProperty -Path "HKCU:\Software\MPC-BE Filters\MPC Video Renderer" -Name "VPSuperResolution" -Value 1
    ```
  * Disable (Value `0`):
    ```powershell
    Set-ItemProperty -Path "HKCU:\Software\MPC-BE Filters\MPC Video Renderer" -Name "VPSuperResolution" -Value 0
    ```
* **Option C — AMD Driver Setting**: Open **AMD Software: Adrenalin Edition** &rarr; **Gaming** &rarr; **Graphics** &rarr; Enable **Video Super Resolution (VSR)**.

---

## 6. Verification & Telemetry

1. **OSD Telemetry**: During playback in MPC-BE, press `Ctrl + J` to inspect the MPC Video Renderer OSD:
   - `Color: RGB 10-bit HDR10: On`
   - `VideoProcessor: D3D11 VP, output to R10G10B10A2_UNORM`
   - `Presentation: Flip discard, R10G10B10A2_UNORM`
2. **GPU Activity**: Check Windows Task Manager (`Ctrl + Shift + Esc`) -> GPU -> Compute:
   - Verify steady 8% to 15% compute activity during playback, dropping to 0% when paused.
3. **Live A/B Testing**: In `%APPDATA%\SVP4\override.js`, comment out the `clip = vs_directml_hdr.Convert(...)` line with `//` and save:
   - SVP reloads live during playback without restarting playback. Notice how contrast flattens and highlights dim. Removing `//` immediately restores punchy HDR depth.

---

## 7. Project Structure

```text
OpenHDR-DirectML/
├── models_arch.py             # PyTorch SDR-to-HDR neural architecture
├── build_cinematic_hdr.py     # Calibrated S-curve ONNX export tool
├── hdrtvnet_1080p_fp16.onnx   # Production DirectML FP16 model
├── vs_directml_hdr.py         # Core VapourSynth filter node implementation
├── openhdr.ini                # Central player configuration file
├── openhdr.vpy                # Standalone script for non-SVP MPC-BE users
├── benchmark_dml.py           # Hardware latency benchmark tool
├── test_pipeline.py           # Verification and pattern generator suite
├── requirements.txt           # Python package dependencies
├── setup_env.ps1              # Automated local venv environment builder
├── install.ps1                # Automated benchmark and setup runner
└── README.md                  # Master documentation
```

---

## 8. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
