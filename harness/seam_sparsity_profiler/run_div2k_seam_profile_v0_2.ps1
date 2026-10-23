# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "harness\seam_sparsity_profiler"
$executable = Join-Path $harness (
    "gpu_seam_sparsity_profiler_v0_1.exe")
$dataset = Join-Path $root (
    "data\" +
    "DIV2K_valid_HR_official\DIV2K_valid_HR")
$result = Join-Path $root (
    "results\" +
    "gpu_seam_sparsity_DIV2K_valid100_q2_q256_v0.2")
$csv = Join-Path $result "seam_sparsity.csv"
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

New-Item -ItemType Directory -Path $result -Force | Out-Null
& $executable `
    $dataset `
    "--output=$csv" `
    "--tiles=16,32,64,128" `
    "--levels=256,32,16,8,4,2" `
    "--device=0" `
    "--block=16" |
    Tee-Object -FilePath (Join-Path $result "console.log")
if ($LASTEXITCODE -ne 0) {
    throw "DIV2K seam profile failed with exit code $LASTEXITCODE"
}

Write-Output "[DIV2K-SEAM-PROFILE-v0.2-OK]"
Write-Output (
    "csv_sha256=" +
    (Get-FileHash -LiteralPath $csv -Algorithm SHA256).Hash)
