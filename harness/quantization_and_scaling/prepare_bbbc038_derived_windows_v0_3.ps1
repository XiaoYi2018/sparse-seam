# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$dataset = Join-Path $root (
    "data\BBBC038_stage1_train_derived_v0.3")
$archive = Join-Path $dataset (
    "transfer\BBBC038_derived_marker_watershed_v0_2.tar.gz")
$derived = Join-Path $dataset "derived_marker_watershed_v0.2"
$windowsTar = "C:\Windows\System32\tar.exe"
$expectedDerivedArchiveSha256 = (
    "80198B911342ECD8DCCE03294E17162FA5080D13B706DE8B0527446B92A754AD")
$officialArchiveSha256 = (
    "DCB6EDC2690F137406638B2309581A71522C4DFF19157D118453B448DCDDCB68")

if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "Transferred derived archive is missing: $archive"
}
$actualSha256 = (
    Get-FileHash -LiteralPath $archive -Algorithm SHA256
).Hash
if ($actualSha256 -ne $expectedDerivedArchiveSha256) {
    throw (
        "Derived archive hash mismatch: expected " +
        "$expectedDerivedArchiveSha256, found $actualSha256")
}
if (Test-Path -LiteralPath $derived) {
    throw "Refusing to overwrite existing derived directory: $derived"
}

& $windowsTar -xf $archive -C $dataset
if ($LASTEXITCODE -ne 0) {
    throw "Windows tar failed with exit code $LASTEXITCODE"
}

$distanceRoot = Join-Path $derived "distance_maps"
$manifest = Join-Path $derived "application_manifest_v0.2.csv"
$distanceCount = @(
    Get-ChildItem -LiteralPath $distanceRoot -File -Filter "*.png"
).Count
$manifestCount = @(Import-Csv -LiteralPath $manifest).Count
if ($distanceCount -ne 670 -or $manifestCount -ne 670) {
    throw (
        "Expected 670 distance maps and rows; found " +
        "$distanceCount and $manifestCount")
}

$record = [PSCustomObject]@{
    dataset = "BBBC038v1 stage1_train derived distance maps"
    source_url = (
        "https://data.broadinstitute.org/bbbc/BBBC038/" +
        "stage1_train.zip")
    license = "CC0"
    official_archive_sha256 = $officialArchiveSha256
    derivation_route = "official_archive_derived_on_linux_host_then_copied_to_windows"
    derived_archive_sha256 = $actualSha256
    distance_map_count = $distanceCount
    manifest_sha256 = (
        Get-FileHash -LiteralPath $manifest -Algorithm SHA256
    ).Hash
    prepared_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$record |
    Export-Csv -LiteralPath (
        Join-Path $dataset "source_record_v0.3.csv") `
        -NoTypeInformation `
        -Encoding UTF8

Write-Output (
    "[BBBC038-DERIVED-WINDOWS-v0.3-OK] " +
    "distance_maps=$distanceCount rows=$manifestCount " +
    "derived_archive_sha256=$actualSha256")
Write-Output (
    "manifest_sha256=$($record.manifest_sha256)")
