# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$dataset = Join-Path $root (
    "data\BSDS500_official_v0.1")
$archive = Join-Path $dataset "download\BSR_bsds500.tgz"
$extracted = Join-Path $dataset "extracted_v0.2"
$images = Join-Path $dataset "images_v0.2"
$sourceUrl = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/" +
    "vision/grouping/BSR/BSR_bsds500.tgz")
$windowsTar = "C:\Windows\System32\tar.exe"

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "Downloaded archive is missing: $archive"
}
if (Test-Path -LiteralPath $extracted) {
    throw "Refusing to overwrite existing extraction: $extracted"
}
if (Test-Path -LiteralPath $images) {
    throw "Refusing to overwrite existing image set: $images"
}
New-Item -ItemType Directory -Path $extracted | Out-Null
New-Item -ItemType Directory -Path $images | Out-Null

& $windowsTar -xf $archive -C $extracted
if ($LASTEXITCODE -ne 0) {
    throw "Windows tar failed with exit code $LASTEXITCODE"
}

$sourceRoot = Join-Path $extracted "BSR\BSDS500\data\images"
$manifest = @()
foreach ($split in @("train", "val", "test")) {
    $splitRoot = Join-Path $sourceRoot $split
    $splitFiles = @(
        Get-ChildItem -LiteralPath $splitRoot -File |
            Where-Object { $_.Extension -ieq ".jpg" } |
            Sort-Object Name
    )
    foreach ($source in $splitFiles) {
        $localName = $split + "_" + $source.Name
        $destination = Join-Path $images $localName
        Copy-Item -LiteralPath $source.FullName -Destination $destination
        $manifest += [PSCustomObject]@{
            local_file = $localName
            split = $split
            original_file = $source.Name
            source_relative_path = (
                "BSR\BSDS500\data\images\" +
                $split + "\" + $source.Name)
            sha256 = (
                Get-FileHash -LiteralPath $destination `
                    -Algorithm SHA256
            ).Hash
        }
    }
}

$manifestPath = Join-Path $dataset "manifest_v0.2.csv"
$manifest |
    Export-Csv -LiteralPath $manifestPath `
        -NoTypeInformation `
        -Encoding UTF8

$sourceRecord = [PSCustomObject]@{
    dataset = "Berkeley Segmentation Data Set and Benchmarks 500"
    source_url = $sourceUrl
    archive_file = $archive
    archive_sha256 = (
        Get-FileHash -LiteralPath $archive -Algorithm SHA256
    ).Hash
    image_directory = $images
    image_count = $manifest.Count
    train_count = @($manifest | Where-Object split -eq "train").Count
    val_count = @($manifest | Where-Object split -eq "val").Count
    test_count = @($manifest | Where-Object split -eq "test").Count
    retrieved_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$sourceRecord |
    Export-Csv -LiteralPath (
        Join-Path $dataset "source_record_v0.2.csv") `
        -NoTypeInformation `
        -Encoding UTF8

if ($manifest.Count -ne 500) {
    throw "Expected 500 BSDS500 images, found $($manifest.Count)"
}
Write-Output (
    "[BSDS500-OFFICIAL-v0.2-OK] images=$($manifest.Count) " +
    "archive_sha256=$($sourceRecord.archive_sha256)")
Write-Output (
    "manifest_sha256=" +
    (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash)
