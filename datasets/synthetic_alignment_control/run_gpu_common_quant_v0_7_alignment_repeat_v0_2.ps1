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
    "data\synthetic_scaling_alignment_v0.2")
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

$orders = @("gos", "sog", "gos", "sog", "gos", "sog")
for ($index = 0; $index -lt $orders.Count; ++$index) {
    $trial = $index + 1
    $order = $orders[$index]
    $resultName = (
        "gpu_common_quant_v0.7_scaling_alignment_v0.2_" +
        "trial{0:D2}_{1}" -f $trial, $order)
    $result = Join-Path $root (
        "results\" + $resultName)
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
        throw (
            "alignment trial $trial order $order failed " +
            "with exit code $LASTEXITCODE")
    }
    Write-Output (
        "[GPU-COMMON-QUANT-v0.7-ALIGNMENT] " +
        "trial=$trial order=$order OK")
}

Write-Output (
    "[GPU-COMMON-QUANT-v0.7-ALIGNMENT-v0.2-OK]")
