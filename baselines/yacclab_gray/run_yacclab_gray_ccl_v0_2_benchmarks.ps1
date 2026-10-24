# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$executable = Join-Path $repo "baselines\yacclab_gray\yacclab_gray_ccl_v0_2.exe"
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"
$datasets = @(
    @{ Name = "DIV2K_valid100"; Inputs = Join-Path $ssRoot "data\DIV2K_valid_HR_official\DIV2K_valid_HR" },
    @{ Name = "BSDS500_v0.2"; Inputs = Join-Path $ssRoot "data\BSDS500_official_v0.1\images_v0.2" },
    @{ Name = "BBBC038_distance_v0.3"; Inputs = Join-Path $ssRoot "data\BBBC038_stage1_train_derived_v0.3\derived_marker_watershed_v0.2\distance_maps" }
)

foreach ($dataset in $datasets) {
    $result = Join-Path $ssRoot (
        "results\yacclab_gray_ccl_v0.2_{0}_w5_r30" -f $dataset.Name)
    & $executable $dataset.Inputs "--output=$result" "--warmup=5" "--reps=30"
    if ($LASTEXITCODE -ne 0) {
        throw "$($dataset.Name) failed with exit code $LASTEXITCODE"
    }
}
Write-Output "[YACCLAB-GRAY-v0.2-BENCHMARKS-OK]"
