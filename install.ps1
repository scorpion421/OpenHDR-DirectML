# OpenHDR-DirectML Setup and Installation Script
# Automated one-click setup for AMD Radeon RX 7900 XTX (DirectML / VapourSynth)
# Keeps all dependencies strictly isolated within this project directory.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptDir

Write-Host "============================================================"
Write-Host "OpenHDR-DirectML: Automated Local Setup"
Write-Host "============================================================"

# 1. Check Python Launcher
$pythonLauncher = "py"
$pythonVersionArg = "-3.13"

try {
    $pyVersion = & $pythonLauncher $pythonVersionArg --version 2>&1
    Write-Host "Found Python: $pyVersion"
} catch {
    $pythonVersionArg = "-3.12"
    try {
        $pyVersion = & $pythonLauncher $pythonVersionArg --version 2>&1
        Write-Host "Found Python: $pyVersion"
    } catch {
        Write-Error "Python 3.13 or 3.12 is required. Please ensure Python is installed."
        exit 1
    }
}

# 2. Virtual Environment Setup
$venvPath = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating local virtual environment in $venvPath..."
    & $pythonLauncher $pythonVersionArg -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to initialize virtual environment."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Local virtual environment already exists at $venvPath."
}

# 3. Dependency Installation
Write-Host "Installing project requirements into local virtual environment..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r (Join-Path $scriptDir "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies from requirements.txt."
    exit $LASTEXITCODE
}

# Install portable VapourSynth wheel if present
$whlFiles = Get-ChildItem (Join-Path $scriptDir "vs_portable\wheel\*.whl") -ErrorAction SilentlyContinue
if ($whlFiles) {
    Write-Host "Installing local VapourSynth wheel into virtual environment..."
    & $venvPython -m pip install $whlFiles[0].FullName --quiet
}

# 4. DirectML Check
Write-Host "Checking DirectML runtime..."
$dmlCheck = & $venvPython -c "import onnxruntime as ort; print('DmlExecutionProvider' in ort.get_available_providers())"
if ($dmlCheck.Trim() -ne "True") {
    Write-Warning "DirectML Execution Provider was not detected. GPU acceleration may not be available."
} else {
    Write-Host "DirectML Execution Provider is active and ready."
}

# 5. Model Export
$modelFile = Join-Path $scriptDir "hdrtvnet_1080p_fp16.onnx"
if (-not (Test-Path $modelFile)) {
    Write-Host "Generating optimized FP16 ONNX model for 1080p..."
    & $venvPython (Join-Path $scriptDir "export_onnx.py") --output $modelFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Model export failed."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "FP16 ONNX model found: $modelFile"
}

# 6. Verification and Benchmark
Write-Host "Running DirectML hardware benchmark on GPU..."
& $venvPython (Join-Path $scriptDir "benchmark_dml.py") --model $modelFile

Write-Host "Running end-to-end pipeline verification..."
& $venvPython (Join-Path $scriptDir "test_pipeline.py") --model $modelFile

Write-Host "============================================================"
Write-Host "OpenHDR-DirectML installation and verification completed successfully!"
Write-Host "============================================================"
