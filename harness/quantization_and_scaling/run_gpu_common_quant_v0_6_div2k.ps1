# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "harness\quantization_and_scaling"
$executable = Join-Path $harness "gpu_common_quant_v0_6.exe"
$dataset = Join-Path $root (
    "data\" +
    "DIV2K_valid_HR_official\DIV2K_valid_HR")
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

foreach ($levels in @(256, 32, 16, 8, 4, 2)) {
    foreach ($order in @("gos", "sog")) {
        $result = Join-Path $root (
            "results\" +
            "gpu_common_quant_v0.6_DIV2K_valid100_q{0}_w5_r30_{1}" -f
            $levels, $order)
        New-Item -ItemType Directory -Path $result -Force | Out-Null
        & $executable `
            $dataset `
            "--output=$result" `
            "--device=0" `
            "--block=16" `
            "--warmup=5" `
            "--reps=30" `
            "--order=$order" `
            "--levels=$levels" |
            Tee-Object -FilePath (Join-Path $result "console.log")
        if ($LASTEXITCODE -ne 0) {
            throw (
                "DIV2K q{0} {1} failed with exit code {2}" -f
                $levels, $order, $LASTEXITCODE)
        }
    }
}

Write-Output "[GPU-COMMON-QUANT-v0.6-DIV2K-OK]"
