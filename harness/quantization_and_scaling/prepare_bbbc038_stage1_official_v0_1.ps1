# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$dataset = Join-Path $root (
    "data\BBBC038_stage1_train_official_v0.1")
$download = Join-Path $dataset "download"
$extracted = Join-Path $dataset "extracted"
$archive = Join-Path $download "stage1_train.zip"
$sourceUrl = (
    "https://data.broadinstitute.org/bbbc/BBBC038/" +
    "stage1_train.zip")
$windowsTar = "C:\Windows\System32\tar.exe"

if (Test-Path -LiteralPath $dataset) {
    throw "Refusing to overwrite existing dataset: $dataset"
}
New-Item -ItemType Directory -Path $download | Out-Null
New-Item -ItemType Directory -Path $extracted | Out-Null

& curl.exe `
    -L `
    --fail `
    --retry 3 `
    --retry-delay 2 `
    --output $archive `
    $sourceUrl
if ($LASTEXITCODE -ne 0) {
    throw "curl failed with exit code $LASTEXITCODE"
}

& $windowsTar -xf $archive -C $extracted
if ($LASTEXITCODE -ne 0) {
    throw "Windows tar failed with exit code $LASTEXITCODE"
}

$sampleCount = @(
    Get-ChildItem -LiteralPath $extracted -Directory -Recurse |
        Where-Object {
            Test-Path -LiteralPath (
                Join-Path $_.FullName "images") -PathType Container
        } |
        Where-Object {
            Test-Path -LiteralPath (
                Join-Path $_.FullName "masks") -PathType Container
        }
).Count
if ($sampleCount -ne 670) {
    throw "Expected 670 stage1 training samples, found $sampleCount"
}

$record = [PSCustomObject]@{
    dataset = "BBBC038v1 Kaggle 2018 Data Science Bowl stage1_train"
    source_url = $sourceUrl
    archive_file = $archive
    archive_sha256 = (
        Get-FileHash -LiteralPath $archive -Algorithm SHA256
    ).Hash
    sample_count = $sampleCount
    license = "CC0"
    retrieved_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$record |
    Export-Csv -LiteralPath (
        Join-Path $dataset "source_record_v0.1.csv") `
        -NoTypeInformation `
        -Encoding UTF8

Write-Output (
    "[BBBC038-STAGE1-v0.1-OK] samples=$sampleCount " +
    "archive_sha256=$($record.archive_sha256)")
