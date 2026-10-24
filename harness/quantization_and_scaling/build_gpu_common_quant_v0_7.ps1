# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "harness\quantization_and_scaling"
$source = Join-Path $harness "gpu_common_quant_v0_7.cu"
$executable = Join-Path $harness "gpu_common_quant_v0_7.exe"
$nvcc = (
    "C:\Program Files\NVIDIA GPU Computing Toolkit\" +
    "CUDA\v12.6\bin\nvcc.exe")
$opencvRoot = "$ocvRoot"

& $nvcc `
    -O3 `
    -std=c++17 `
    -arch=sm_89 `
    -DSS_TILE=32 `
    -DSS_CLASSIFY_VARIANT=1 `
    -Xcompiler "/EHsc /O2" `
    "-I$(Join-Path $opencvRoot 'include')" `
    "-L$(Join-Path $opencvRoot 'x64\vc16\lib')" `
    -lopencv_world4110 `
    $source `
    -o $executable
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Output "[GPU-COMMON-QUANT-v0.7-BUILD-OK]"
Write-Output (
    "source_sha256=" +
    (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash)
Write-Output (
    "executable_sha256=" +
    (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash)
