# Setup script for isolated local Python virtual environment
# Project: OpenHDR-DirectML
# Ensures no global system packages or settings are modified.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $scriptDir

Write-Host "Setting up isolated virtual environment in: $scriptDir\.venv"

# Detect compatible Python launcher (Python 3.13 or 3.12)
$pythonLauncher = "py"
$pythonVersionArg = "-3.13"

try {
    & $pythonLauncher $pythonVersionArg --version
} catch {
    $pythonVersionArg = "-3.12"
    try {
        & $pythonLauncher $pythonVersionArg --version
    } catch {
        Write-Error "Python 3.13 or 3.12 is required for DirectML and modern VapourSynth ABI support."
        exit 1
    }
}

$venvPath = Join-Path $scriptDir ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $venvPath using $pythonVersionArg..."
    & $pythonLauncher $pythonVersionArg -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Existing virtual environment found."
}

Write-Host "Upgrading pip inside local virtual environment..."
& $venvPython -m pip install --upgrade pip --quiet

Write-Host "Installing requirements from requirements.txt into local virtual environment..."
& $venvPython -m pip install -r (Join-Path $scriptDir "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies from requirements.txt."
    exit $LASTEXITCODE
}

# Install portable VapourSynth wheel if present
$whlFiles = Get-ChildItem (Join-Path $scriptDir "vs_portable\wheel\*.whl") -ErrorAction SilentlyContinue
if ($whlFiles) {
    Write-Host "Installing local VapourSynth wheel..."
    & $venvPython -m pip install $whlFiles[0].FullName --quiet
}

Write-Host "Verifying DirectML execution provider and VapourSynth..."
$verifyScript = @"
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available ONNX Runtime Providers:', providers)
if 'DmlExecutionProvider' in providers:
    print('DirectML Execution Provider is successfully registered.')
else:
    print('WARNING: DmlExecutionProvider not found in available providers.')

try:
    import vapoursynth as vs
    print('VapourSynth version:', str(vs.core))
except ImportError as e:
    print('VapourSynth not loaded:', e)
"@

& $venvPython -c $verifyScript
Write-Host "Local environment setup complete."
