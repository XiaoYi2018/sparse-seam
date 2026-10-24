# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = "$ssRoot"
$harness = Join-Path $repo "baselines\iwpp"
$source = Join-Path $harness "iwpp_reconstruction_cuda_v0_1.cu"
$executable = Join-Path $harness "iwpp_reconstruction_cuda_v0_1.exe"
$dataset = Join-Path $root "data\DIV2K_valid_HR_official\DIV2K_valid_HR"
$result = Join-Path $root "results\iwpp_reconstruction_cuda_v0.1_DIV2K_valid100_30rep_v0.2"
$opencvBin = "$ocvRoot\x64\vc16\bin"
$wrapper = $MyInvocation.MyCommand.Path
$expectedSourceHash = "C17E2561A638AA21DCC16D62121CF6ADC30E4B8F1D66915D4CFCACD465CC0CD5"

if (Test-Path -LiteralPath $result) {
    throw "Refusing to overwrite existing result directory: $result"
}
foreach ($required in @($source, $executable, $dataset, $opencvBin)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required path: $required"
    }
}

$images = @(Get-ChildItem -LiteralPath $dataset -File | Sort-Object Name)
if ($images.Count -ne 100) {
    throw "Expected exactly 100 DIV2K images, found $($images.Count)"
}
if (@($images.Name | Select-Object -Unique).Count -ne 100) {
    throw "DIV2K filenames are not unique"
}

$sourceHashBefore = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
$exeHashBefore = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
$wrapperHashBefore = (Get-FileHash -LiteralPath $wrapper -Algorithm SHA256).Hash
if ($sourceHashBefore -ne $expectedSourceHash) {
    throw "Source identity mismatch: $sourceHashBefore"
}
# The executable hash depends on the local build and is recorded, not checked.

New-Item -ItemType Directory -Path $result | Out-Null
$manifest = Join-Path $result "dataset_manifest_sha256.csv"
"name,length,sha256" | Set-Content -LiteralPath $manifest -Encoding UTF8
foreach ($image in $images) {
    $hash = (Get-FileHash -LiteralPath $image.FullName -Algorithm SHA256).Hash
    ('"{0}",{1},{2}' -f $image.Name, $image.Length, $hash) |
        Add-Content -LiteralPath $manifest -Encoding UTF8
}
$manifestHash = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash

$env:PATH = "$opencvBin;$env:PATH"
$env:CUDA_MODULE_LOADING = "EAGER"
& $executable `
    $dataset `
    "--output=$result" `
    "--device=0" `
    "--warmup=5" `
    "--reps=30" |
    Tee-Object -FilePath (Join-Path $result "console.log")
if ($LASTEXITCODE -ne 0) {
    throw "IWPP exited with code $LASTEXITCODE"
}

$csvPath = Join-Path $result "iwpp_reconstruction_v0_1.csv"
$rows = @(Import-Csv -LiteralPath $csvPath)
if ($rows.Count -ne 100) {
    throw "Expected 100 result rows, found $($rows.Count)"
}
if (@($rows.image | Select-Object -Unique).Count -ne 100) {
    throw "Result image names are not unique"
}
$inputNames = @($images.Name | Sort-Object)
$resultNames = @($rows.image | Sort-Object)
if (Compare-Object -ReferenceObject $inputNames -DifferenceObject $resultNames) {
    throw "Result image-name set does not match the input manifest"
}
if (@($rows | Where-Object { [int]$_.warmup -ne 5 -or [int]$_.repetitions -ne 30 }).Count -ne 0) {
    throw "Result metadata does not report 5 warm-ups and 30 repetitions"
}
foreach ($row in $rows) {
    foreach ($field in @("width", "height", "megapixels", "median_compute_ms")) {
        $value = [double]$row.$field
        if ($value -le 0 -or [double]::IsNaN($value) -or [double]::IsInfinity($value)) {
            throw "Invalid $field for image $($row.image): $value"
        }
    }
}

$totalMegapixels = ($rows | Measure-Object -Property megapixels -Sum).Sum
$totalMedianMs = ($rows | Measure-Object -Property median_compute_ms -Sum).Sum
if ([double]$totalMegapixels -le 0 -or [double]$totalMedianMs -le 0 -or
    [double]::IsNaN([double]$totalMegapixels) -or
    [double]::IsNaN([double]$totalMedianMs) -or
    [double]::IsInfinity([double]$totalMegapixels) -or
    [double]::IsInfinity([double]$totalMedianMs)) {
    throw "Invalid aggregate inputs"
}
$aggregateMpixPerSecond = 1000.0 * [double]$totalMegapixels / [double]$totalMedianMs

$sourceHashAfter = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
$exeHashAfter = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
$wrapperHashAfter = (Get-FileHash -LiteralPath $wrapper -Algorithm SHA256).Hash
if ($sourceHashBefore -ne $sourceHashAfter -or $exeHashBefore -ne $exeHashAfter -or
    $wrapperHashBefore -ne $wrapperHashAfter) {
    throw "Source, executable, or wrapper hash changed during the run"
}
$uncheckedMaxSum = ($rows | Measure-Object -Property oracle_max_mismatches -Sum).Sum
$uncheckedMinSum = ($rows | Measure-Object -Property oracle_min_mismatches -Sum).Sum
$csvHash = (Get-FileHash -LiteralPath $csvPath -Algorithm SHA256).Hash
$environmentPath = Join-Path $result "environment.txt"
$consolePath = Join-Path $result "console.log"
$environmentHash = (Get-FileHash -LiteralPath $environmentPath -Algorithm SHA256).Hash
$consoleHash = (Get-FileHash -LiteralPath $consolePath -Algorithm SHA256).Hash

$runManifest = @(
    "experiment=IWPP DIV2K 5-warmup 30-repetition rerun v0.2"
    "expected_max_dir_provided=false"
    "expected_min_dir_provided=false"
    "oracle_checked=false"
    "zero_mismatch_fields_are_not_oracle_evidence=true"
    "frozen_executable_OK_marker_is_not_oracle_evidence=true"
    "device=0"
    "dataset_path=$dataset"
    "result_path=$result"
    "command=$executable $dataset --output=$result --device=0 --warmup=5 --reps=30"
    "cuda_module_loading=EAGER"
    "dataset_count=100"
    "dataset_manifest_sha256=$manifestHash"
    "source_sha256=$sourceHashAfter"
    "executable_sha256=$exeHashAfter"
    "wrapper_sha256=$wrapperHashAfter"
    "csv_sha256=$csvHash"
    "environment_sha256=$environmentHash"
    "console_sha256=$consoleHash"
    "unchecked_oracle_max_mismatch_field_sum=$uncheckedMaxSum"
    "unchecked_oracle_min_mismatch_field_sum=$uncheckedMinSum"
    "warmup=5"
    "repetitions=30"
    ("total_megapixels={0:F6}" -f [double]$totalMegapixels)
    ("sum_per_image_median_ms={0:F6}" -f [double]$totalMedianMs)
    ("aggregate_compute_mpix_s={0:F6}" -f $aggregateMpixPerSecond)
)
$runManifest | Set-Content -LiteralPath (Join-Path $result "run_manifest.txt") -Encoding UTF8

$artifactHashes = @(
    "artifact,sha256"
    ('"source","{0}"' -f $sourceHashAfter)
    ('"executable","{0}"' -f $exeHashAfter)
    ('"wrapper","{0}"' -f $wrapperHashAfter)
    ('"dataset_manifest","{0}"' -f $manifestHash)
    ('"result_csv","{0}"' -f $csvHash)
    ('"environment","{0}"' -f $environmentHash)
    ('"console","{0}"' -f $consoleHash)
)
$artifactHashes | Set-Content -LiteralPath (Join-Path $result "artifact_hashes.csv") -Encoding UTF8

Write-Output "[IWPP-DIV2K-30REP-v0.2-MEASUREMENT-OK-NO-ORACLE]"
Write-Output ("aggregate_compute_mpix_s={0:F6}" -f $aggregateMpixPerSecond)
Write-Output "oracle_checked=false"
Write-Output "source_sha256=$sourceHashAfter"
Write-Output "executable_sha256=$exeHashAfter"
Write-Output "dataset_manifest_sha256=$manifestHash"
