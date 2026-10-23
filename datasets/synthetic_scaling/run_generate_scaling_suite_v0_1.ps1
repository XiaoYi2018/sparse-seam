# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "datasets\synthetic_scaling"
$executable = Join-Path $harness (
    "generate_scaling_suite_v0_1.exe")
$output = Join-Path $root (
    "data\synthetic_scaling_v0.1")

New-Item -ItemType Directory -Path $output -Force | Out-Null
& $executable $output
if ($LASTEXITCODE -ne 0) {
    throw "generator failed with exit code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $output -File |
    Get-FileHash -Algorithm SHA256 |
    Select-Object Hash, Path |
    Export-Csv -LiteralPath (
        Join-Path $output "SHA256SUMS_v0.1.csv") `
        -NoTypeInformation `
        -Encoding UTF8

Write-Output "[SCALING-SUITE-v0.1-OK]"
