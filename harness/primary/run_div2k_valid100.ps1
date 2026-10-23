# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = "$ssRoot"
$executable = Join-Path $repo "harness\primary\gpu_common_v0_5.exe"
$dataset = Join-Path $root (
    "data\DIV2K_valid_HR_official\DIV2K_valid_HR")
$env:PATH = "$ocvRoot\x64\vc16\bin;$env:PATH"

foreach ($order in @("gos", "sog")) {
    $result = Join-Path $root (
        "results\" +
        "gpu_common_v0.5_DIV2K_valid100_30rep_{0}" -f $order)
    New-Item -ItemType Directory -Path $result -Force | Out-Null
    & $executable `
        $dataset `
        "--output=$result" `
        "--device=0" `
        "--block=16" `
        "--warmup=5" `
        "--reps=30" `
        "--order=$order" |
        Tee-Object -FilePath (Join-Path $result "console.log")
    if ($LASTEXITCODE -ne 0) {
        throw "DIV2K $order benchmark failed: $LASTEXITCODE"
    }
}

Write-Output "[GPU-COMMON-v0.5-DIV2K-VALID100-OK]"
