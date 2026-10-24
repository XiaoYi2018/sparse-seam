# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "harness\quantization_and_scaling"
$executable = Join-Path $harness "gpu_common_quant_v0_7.exe"
$dataset = Join-Path $root (
    "data\BBBC038_stage1_train_derived_v0.3\" +
    "derived_marker_watershed_v0.2\distance_maps")
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

foreach ($order in @("gos", "sog")) {
    $result = Join-Path $root (
        "results\" +
        "gpu_common_quant_v0.7_BBBC038_distance_v0.3_" +
        "q256_w5_r30_{0}" -f $order)
    if (Test-Path -LiteralPath $result) {
        throw "Refusing to overwrite existing result: $result"
    }
    New-Item -ItemType Directory -Path $result | Out-Null
    & $executable `
        $dataset `
        "--output=$result" `
        "--device=0" `
        "--block=16" `
        "--warmup=5" `
        "--reps=30" `
        "--order=$order" `
        "--levels=256" |
        Tee-Object -FilePath (Join-Path $result "console.log")
    if ($LASTEXITCODE -ne 0) {
        throw "BBBC038 distance $order failed: $LASTEXITCODE"
    }
}

Write-Output (
    "[GPU-COMMON-QUANT-v0.7-BBBC038-DISTANCE-v0.3-OK]")
