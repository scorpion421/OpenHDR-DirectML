# Project Blueprint: OpenHDR-DirectML
**Real-Time AI-Powered SDR-to-HDR Video Upconversion for AMD Radeon (RDNA 3)**

---

## 1. Project Overview & Objective
Establish an open-source, hardware-agnostic AI video pipeline that performs real-time **8-bit SDR (Rec.709)** to **10-bit HDR10 (BT.2020 / SMPTE ST 2084 PQ)** conversion on modern AMD Radeon GPUs (specifically optimized for the **AMD Radeon RX 7900 XTX**).

* **Target Hardware:** AMD Radeon RX 7000 Series (RDNA 3 AI Matrix Accelerators) via **Microsoft DirectML (DirectX 12)**.
* **Media Player Stack:** **MPC-BE** / **MPC-HC** utilizing **AviSynth Filter (AVSF)** and **VapourSynth**.
* **Coexistence:** Seamlessly integrates with **SVP 4 (SmoothVideo Project)** and **MPC Video Renderer (MPCVR)**.

---

## 2. Complete End-to-End Signal & Pipeline Architecture

```text
[Source Media: 1080p SDR 8-bit Rec.709 @ 24fps]
                        │
                        ▼
       Hardware Video Decoder (LAV / MPC Decoder)
            (Outputs NV12 / YUV420P8)
                        │
                        ▼
          AviSynth Filter (AVSF) in MPC-BE
                        │
                        ▼
             VapourSynth Core Engine
                        │
  ┌─────────────────────┴──────────────────────────────┐
  │  STAGE 1: OpenHDR-DirectML Filter Node             │
  │  1. Ingest frame & convert to FP16 Tensor (NCHW)   │
  │  2. DirectML Inference on RX 7900 XTX (~4 ms)      │
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
               BenQ MOBIUZ EX3415R
    (Native DisplayHDR 400 + MOBIUZ_Ultimate_HDR)
```

---

## 3. The Neural Network Model

### Primary Candidate: HDRTVNet
* **Reference:** CVPR / Shanghai Jiao Tong University research for television broadcast SDR-to-HDR conversion.
* **Multi-Branch Architecture:**
  1. **Global Color Mapping (GCM):** Expands the color gamut from Rec.709 to DCI-P3 / BT.2020 without hue drift.
  2. **Local Tone Mapping (LTM):** Dynamically adjusts scene midtones while preserving natural human skin tones (100–140 Nits).
  3. **Highlight Enhancement (HE):** Selectively reconstructs overexposed/specular areas (lamps, headlights, sun reflections) and boosts them to 400–1000 Nits.

### Alternative (Ultra-Low Latency): AdaInt (Adaptive 3D-LUT)
* A lightweight convolutional network predicts coefficients for a dynamic 33x33x33 3D-LUT per frame.
* Latency: **<1.5 ms** per 1080p frame on the RX 7900 XTX.

### Model Optimization for RDNA 3:
* **Format:** Optimized **ONNX FP16** (Half Precision).
* **Benefits:** Utilizes Dual-Issue SIMD / AI Matrix Accelerators on Navi 31, cuts VRAM footprint in half, doubles throughput.
* **Shape:** Fixed batch dimension `[1, 3, 1080, 1920]` to eliminate runtime allocation stalls.

---

## 4. Execution Engine: Microsoft DirectML

* **Package:** `onnxruntime-directml`
* **DirectX 12 Compute:** Runs natively across any modern GPU architecture (AMD, Intel, NVIDIA).
* **Python Initialization Pattern:**
  ```python
  import onnxruntime as ort

  session_options = ort.SessionOptions()
  session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  session_options.enable_mem_pattern = True

  # Device 0 targets primary discrete GPU (AMD Radeon RX 7900 XTX)
  session = ort.InferenceSession(
      "hdrtvnet_1080p_fp16.onnx",
      session_options,
      providers=["DmlExecutionProvider"],
      provider_options=[{"device_id": 0}]
  )
  ```

---

## 5. VapourSynth Filter Node Implementation

Implemented as a lightweight Python filter module (`vs_directml_hdr.py`):

```python
import vapoursynth as vs
import numpy as np
import onnxruntime as ort

core = vs.core

class DirectML_HDR:
    def __init__(self, model_path: str, device_id: int = 0):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            opts,
            providers=["DmlExecutionProvider"],
            provider_options=[{"device_id": device_id}]
        )
        self.input_name = self.session.get_inputs()[0].name

    def process_frame(self, n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        # 1. Convert frame to FP16 NCHW tensor
        # 2. Run DirectML Inference
        # 3. Inject HDR10 metadata flags:
        #    _Matrix = 9 (BT.2020)
        #    _Primaries = 9 (BT.2020)
        #    _Transfer = 16 (SMPTE ST 2084 / PQ)
        #    _ColorRange = 1 (Limited)
        # 4. Return 10-bit YUV420P10 frame
        pass
```

---

## 6. SVP 4 (SmoothVideo Project) Coexistence

In the SVP VapourSynth processing script, the AI HDR stage is inserted **before** motion interpolation:

```python
# 1. Ingest 8-bit SDR stream from DirectShow
clip = VpsFilterSource.std.Trim(length=5000000)

# 2. Run DirectML AI HDR conversion at native framerate (24 fps = 41.6 ms budget per frame)
clip = vs_directml_hdr.Convert(clip, model="hdrtvnet_1080p_fp16.onnx")

# 3. SVP interpolates the resulting 10-bit HDR stream to 120/144 fps
smooth = core.svp2.SmoothFps(clip, ...)
smooth.set_output()
```

* **Efficiency Advantage:** The AI only computes **24 frames per second** (utilizing ~10–15% of the RX 7900 XTX), completely avoiding the massive computational overhead of inferencing at 120 fps.

---

## 7. The 5-Phase Implementation Roadmap

### Phase 1: DirectML Hardware Spike & Benchmarking
* Set up a standalone Python environment with `onnxruntime-directml`.
* Run test tensors `(1, 3, 1080, 1920)` through DirectML on the RX 7900 XTX to confirm the 4–6 ms latency target.

### Phase 2: Model Sourcing & FP16 ONNX Export
* Download pretrained open-source checkpoints (HDRTVNet / AdaInt).
* Export and optimize the ONNX computational graph in FP16 precision.
* Validate color correctness against known HDR reference images.

### Phase 3: VapourSynth Filter & Frame-Property Handshake
* Implement `vs_directml_hdr.py`.
* Ensure proper 10-bit frame formatting and injection of `_Transfer=16` / `_Primaries=9` metadata.

### Phase 4: AVSF / MPC-BE Playback Testing
* Play back 1080p SDR video files in MPC-BE.
* Confirm that MPC Video Renderer automatically switches from SDR mode into native 10-bit HDR10 passthrough.
* Test simultaneous execution with SVP 120 fps interpolation.

### Phase 5: Packaging & Open-Source Release
* Initialize dedicated GitHub repository: `scorpion421/DirectML-Video-HDR`.
* Provide an automated PowerShell installation script (`install.ps1`).
* Publish master `README.md` with visual comparison benchmarks and setup instructions.
