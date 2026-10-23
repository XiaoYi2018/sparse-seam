# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "baselines\npp_exact_repair"
$executable = Join-Path $harness "npp_labelmarkersuf_exact_repair_v0_3.exe"
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

$datasets = @(
    @{
        Name = "div2k_validation"
        Inputs = Join-Path $root (
            "data\DIV2K_valid_HR_official\DIV2K_valid_HR")
    }
)

foreach ($dataset in $datasets) {
    $result = Join-Path $root (
        "results\npp_labelmarkersuf_exact_repair_v0.3_{0}_w5_r30" -f
        $dataset.Name)
    New-Item -ItemType Directory -Path $result -Force | Out-Null
    & $executable `
        $dataset.Inputs `
        "--output=$result" `
        "--device=0" `
        "--warmup=5" `
        "--reps=30" |
        Tee-Object -FilePath (Join-Path $result "console.log")
    if ($LASTEXITCODE -ne 0) {
        throw "$($dataset.Name) failed with exit code $LASTEXITCODE"
    }
}

Write-Output "[NPP-UF-REPAIR-v0.3-BENCHMARKS-OK]"
