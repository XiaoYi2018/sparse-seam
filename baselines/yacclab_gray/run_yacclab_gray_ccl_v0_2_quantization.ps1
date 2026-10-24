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
$dataset = Join-Path $ssRoot "data\DIV2K_valid_HR_official\DIV2K_valid_HR"
foreach ($levels in @(256, 32, 16, 8, 4, 2)) {
    $result = Join-Path $ssRoot (
        "results\yacclab_gray_ccl_v0.2_DIV2K_valid100_q{0}_w5_r30" -f $levels)
    & $executable $dataset "--output=$result" "--warmup=5" "--reps=30" "--levels=$levels"
    if ($LASTEXITCODE -ne 0) {
        throw "q$levels failed with exit code $LASTEXITCODE"
    }
}
Write-Output "[YACCLAB-GRAY-v0.2-QUANTIZATION-OK]"
