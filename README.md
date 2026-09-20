# OpenHDR-DirectML
**Real-Time AI-Powered SDR-to-HDR Video Upconversion for AMD Radeon (RDNA 3)**

OpenHDR-DirectML is a hardware-accelerated video enhancement pipeline designed to convert standard 8-bit SDR (Rec.709) video streams into 10-bit HDR10 (BT.2020 / SMPTE ST 2084 PQ) in real time on AMD Radeon GPUs using **Microsoft DirectML (DirectX 12)** and **VapourSynth**.

It seamlessly integrates into modern media player stacks (**MPC-BE**, **AviSynth Filter / AVSF**, **MPC Video Renderer**) and coexists with **SVP 4 (SmoothVideo Project)** motion interpolation.

---

## 1. Pipeline Architecture & Signal Flow

```text
[Source Media: 1080p SDR 8-bit Rec.709 @ 24fps]
                        │
                        ▼
      Hardware Video Decoder (LAV / MPC Decoder)
           (Outputs NV12 / YUV420P8 / RGB24)
                        │
                        ▼
    AviSynth Filter (AVSF) / VapourSynth Filter
                        │
                        ▼
             VapourSynth Core Engine
                        │
  ┌─────────────────────┴──────────────────────────────┐
  │  STAGE 1: OpenHDR-DirectML Filter Node             │
  │  1. Ingest frame & convert to FP16 Tensor (NCHW)   │
  │  2. DirectML Inference on RX 7900 XTX (~5.4 ms)    │
  │  3. Semantic Highlight & Gamut Expansion           │
  │  4. Set VapourSynth HDR Frame Properties:          │
  │     _Matrix = 9, _Primaries = 9, _Transfer = 16    │
  │  5. Output 10-bit YUV420P10 frame buffer           │
  └─────────────────────┬──────────────────────────────┘
                        │ (Stream is now true 10-bit HDR10 @ 24fps)
                        ▼
  ┌────────────────────────────────────────────────────┐
  │  STAGE 2: SVP 4 Motion Interpolation               │
  │  - Takes the 24fps HDR10 stream                    │
  │  - Calculates motion vectors (core.svp2.SmoothFps) │
  │  - Outputs smooth 120 / 144 fps HDR10              │
  └─────────────────────┬──────────────────────────────┘
                        │
                        ▼
             MPC Video Renderer (MPCVR)
      (Automatically receives BT.2020 PQ metadata)
                        │
                        ▼
         DirectX 11 / 12 HDR10 SwapChain
                        │
                        ▼
              HDR Display / Monitor
       (e.g., BenQ MOBIUZ EX3415R DisplayHDR 400)
```

---

## 2. Hardware Benchmarking on AMD Radeon RX 7900 XTX

Benchmarked on **AMD Radeon RX 7900 XTX** (Navi 31, RDNA 3 AI Matrix Accelerators) running FP16 tensor inference at fixed 1080p resolution `[1, 3, 1080, 1920]`:

| Metric | Measured Result | Frame Budget Target (24 fps) |
| :--- | :--- | :--- |
| **Mean Latency** | **5.40 ms** | 41.67 ms |
| **Median Latency** | **5.37 ms** | 41.67 ms |
| **Min / Max Latency** | 5.24 ms / 5.86 ms | - |
| **95th Percentile** | 5.67 ms | - |
| **Max Throughput** | **185.2 FPS** | 24 FPS |
| **GPU Frame Budget Utilization** | **13.0%** | < 25% |

### Key Advantage: Native 24 fps Inference
Because the OpenHDR node processes video **before** SVP motion interpolation, the neural network computes only **24 frames per second**, keeping GPU load at just ~13% and leaving ample compute resources for SVP 4 high-refresh (120/144 fps) interpolation and MPC Video Renderer shaders.

---

## 3. Project Structure

```text
c:\Users\Claymore\Desktop\SDRTOHDR\
├── .venv\                     # Isolated local Python virtual environment
├── models_arch.py             # PyTorch SDR-to-HDR neural network architecture
├── export_onnx.py             # FP16 ONNX export & graph optimization tool
├── hdrtvnet_1080p_fp16.onnx   # Optimized DirectML FP16 model (1080p fixed shape)
├── vs_directml_hdr.py         # Core VapourSynth filter node implementation
├── svp_pipeline.py            # Integration script for SVP 4 + MPC-BE
├── benchmark_dml.py           # Hardware latency benchmark script
├── test_pipeline.py           # End-to-end automated verification suite
├── test_output\               # Generated SDR vs tone-mapped HDR test patterns
├── requirements.txt           # Pinned dependencies
├── setup_env.ps1              # Local virtual environment setup script
├── install.ps1                # One-click automated setup and benchmark runner
└── README.md                  # Master documentation
```

---

## 4. Setup & Installation

All dependencies and scripts are completely self-contained in this directory and do not modify system-wide configurations.

### 1. Automated Setup
Run the PowerShell setup script:
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### 2. Manual Setup
```powershell
# Create virtual environment using Python 3.10
py -3.10 -m venv .venv

# Install dependencies into local environment
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Export FP16 ONNX model
.\.venv\Scripts\python.exe export_onnx.py --output hdrtvnet_1080p_fp16.onnx

# Run benchmark
.\.venv\Scripts\python.exe benchmark_dml.py --model hdrtvnet_1080p_fp16.onnx

# Run verification tests
.\.venv\Scripts\python.exe test_pipeline.py --model hdrtvnet_1080p_fp16.onnx
```

---

## 5. Media Player Integration

### MPC-BE + AviSynth Filter (AVSF) + MPC Video Renderer
1. Open **MPC-BE** -> **Options** -> **External Filters**.
2. Add **VapourSynth Filter** (from `C:\Program Files (x86)\SVP 4\avsf\vapoursynth_filter_64.ax`) and set it to **Prefer**.
3. Set **Video Renderer** to **MPC Video Renderer (MPCVR)**.
4. In MPCVR settings, enable **Passthrough HDR to display**.
5. In the VapourSynth filter script or SVP 4 custom profile, invoke:
   ```python
   import sys
   sys.path.insert(0, r"C:\Users\Claymore\Desktop\SDRTOHDR")
   import vs_directml_hdr

   clip = VpsFilterSource
   clip = vs_directml_hdr.Convert(clip, model_path=r"C:\Users\Claymore\Desktop\SDRTOHDR\hdrtvnet_1080p_fp16.onnx", device_id=0)
   clip.set_output()
   ```

### SVP 4 Coexistence
To combine with SVP 4 motion interpolation, load `svp_pipeline.py`. The stream is converted to 10-bit HDR10 at 24 fps, and SVP then interpolates the HDR10 stream to your monitor refresh rate (e.g. 120 / 144 Hz).

---

## 6. Verification Artifacts

The test suite generates side-by-side verification outputs in `test_output/`:
- `test_sdr_input.png`: Synthetic 1080p Rec.709 color bar, gradient, and highlight pattern.
- `test_hdr_preview.png`: Tone-mapped preview of the output HDR10 signal demonstrating specular highlight expansion and wide gamut mapping.
