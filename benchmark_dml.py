"""DirectML hardware benchmark and latency validation script.

Tests FP16 tensor inference latency on AMD Radeon RX 7900 XTX
via Microsoft DirectML (DirectX 12 Compute).
Target latency: 4-6 ms per 1080p frame (well within 24 fps budget of 41.67 ms).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DirectML inference latency on GPU")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hdrtvnet_1080p_fp16.onnx"),
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU Device ID (0 = primary discrete GPU: AMD Radeon RX 7900 XTX)",
    )
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    return parser.parse_args()


def benchmark_directml(model_path: Path, device_id: int = 0, warmup: int = 15, runs: int = 100) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    providers = ort.get_available_providers()
    print("============================================================")
    print("OpenHDR-DirectML Hardware Benchmark Spike")
    print("============================================================")
    print(f"Available ONNX Runtime Providers: {providers}")

    if "DmlExecutionProvider" not in providers:
        print("ERROR: 'DmlExecutionProvider' is not available. Please verify onnxruntime-directml.")
        return

    # Session options optimized for RDNA 3 AI accelerators
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    provider_options = [{"device_id": device_id}]

    print(f"Initializing DirectML InferenceSession on GPU device_id={device_id}...")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["DmlExecutionProvider"],
        provider_options=provider_options,
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name
    shape = input_meta.shape
    dtype = input_meta.type

    print(f"Model loaded: {model_path.name}")
    print(f"  Input name:  {input_name} (type: {dtype}, shape: {shape})")
    print(f"  Output name: {output_name} (type: {output_meta.type}, shape: {output_meta.shape})")

    np_dtype = np.float16 if "float16" in dtype else np.float32

    # Create dummy frame input (1, 3, 1080, 1920)
    dummy_input = np.random.uniform(0.0, 1.0, size=shape).astype(np_dtype)
    feed_dict = {input_name: dummy_input}

    print(f"\nWarming up DirectML pipeline ({warmup} iterations)...")
    for _ in range(warmup):
        _ = session.run([output_name], feed_dict)

    print(f"Executing benchmark ({runs} iterations)...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], feed_dict)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    fps = 1000.0 / mean_lat

    print("\n---------------- Benchmark Results ----------------")
    print(f"Mean Latency:        {mean_lat:.2f} ms")
    print(f"Median Latency:      {median_lat:.2f} ms")
    print(f"Min / Max Latency:   {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"95th Percentile:     {p95_lat:.2f} ms")
    print(f"99th Percentile:     {p99_lat:.2f} ms")
    print(f"Max Throughput:      {fps:.1f} FPS")
    print("---------------------------------------------------")

    # 24 fps frame budget is 41.67 ms. Target latency is 4-6 ms (~10-15% GPU capacity).
    budget_ms = 1000.0 / 24.0
    load_pct = (mean_lat / budget_ms) * 100.0
    print(f"24 fps Frame Budget: {budget_ms:.2f} ms")
    print(f"GPU Frame Budget Utilization: {load_pct:.1f}%")

    if mean_lat <= 6.0:
        print("\nSUCCESS: Target latency of <= 6.0 ms ACHIEVED!")
    elif mean_lat <= budget_ms:
        print(f"\nACCEPTABLE: Real-time capable at 24 fps (Mean: {mean_lat:.2f} ms < {budget_ms:.2f} ms)")
    else:
        print(f"\nWARNING: Latency ({mean_lat:.2f} ms) exceeds real-time frame budget.")


def main() -> None:
    args = parse_args()
    benchmark_directml(
        model_path=args.model,
        device_id=args.device_id,
        warmup=args.warmup,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
