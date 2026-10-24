# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$harness = Join-Path $repo "datasets\synthetic_alignment_control"
$source = Join-Path $harness "generate_scaling_suite_v0_2.cu"
$executable = Join-Path $harness "generate_scaling_suite_v0_2.exe"
$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"

& $nvcc `
    -O3 `
    -std=c++17 `
    -arch=sm_89 `
    -Xcompiler "/EHsc /O2" `
    $source `
    -o $executable
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Output "[ALIGNMENT-GENERATOR-v0.2-BUILD-OK]"
Write-Output "source_sha256=$((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash)"
Write-Output "executable_sha256=$((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash)"
