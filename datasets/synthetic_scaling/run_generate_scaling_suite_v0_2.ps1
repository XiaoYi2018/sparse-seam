# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$dataset = Join-Path $root (
    "data\synthetic_scaling_v0.1")
$checksum = Join-Path $dataset "SHA256SUMS_v0.2.csv"

$files = Get-ChildItem -LiteralPath $dataset -File |
    Where-Object {
        $_.Name -notlike "SHA256SUMS_*.csv"
    }

$hashes = $files |
    Get-FileHash -Algorithm SHA256 |
    Select-Object Hash, Path

$hashes |
    Export-Csv -LiteralPath $checksum `
        -NoTypeInformation `
        -Encoding UTF8

$pgmCount = @(
    Get-ChildItem -LiteralPath $dataset -File -Filter "*.pgm"
).Count
$manifestCount = @(
    Import-Csv -LiteralPath (Join-Path $dataset "manifest.csv")
).Count
$checksumCount = @(
    Import-Csv -LiteralPath $checksum
).Count

if ($pgmCount -ne 18) {
    throw "Expected 18 PGM files, found $pgmCount"
}
if ($manifestCount -ne 18) {
    throw "Expected 18 manifest rows, found $manifestCount"
}
if ($checksumCount -ne 19) {
    throw (
        "Expected 19 checksums (18 PGM + manifest), found " +
        $checksumCount)
}

Write-Output (
    "[SCALING-SUITE-v0.2-VERIFY-OK] " +
    "pgm=$pgmCount manifest=$manifestCount " +
    "checksums=$checksumCount")
Write-Output (
    "checksum_sha256=" +
    (Get-FileHash -LiteralPath $checksum -Algorithm SHA256).Hash)
