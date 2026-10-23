# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "baselines\blin_dual_tree"
$executable = Join-Path $harness "blin_optim1d_relaxed_flatten_extrema_v0_6_0.exe"
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"

$suites = @(
    @{
        Name = "oracle34"
        Root = Join-Path $repo "oracles\oracle_v0_1"
        Inputs = "inputs_png"
    },
    @{
        Name = "fuzz160"
        Root = Join-Path $repo "oracles\oracle_v0_2_fuzz"
        Inputs = "inputs"
    },
    @{
        Name = "seam12"
        Root = Join-Path $repo "oracles\oracle_v0_3_sparse_seam_adversarial"
        Inputs = "inputs"
    }
)

foreach ($suite in $suites) {
    $result = Join-Path $root (
        "results\" +
        "blin_optim1d_relaxed_flatten_extrema_v0.6.0_{0}_correctness" -f
        $suite.Name)
    New-Item -ItemType Directory -Path $result -Force | Out-Null
    & $executable `
        (Join-Path $suite.Root $suite.Inputs) `
        "--output=$result" `
        "--expected-max=$(Join-Path $suite.Root 'expected_max')" `
        "--expected-min=$(Join-Path $suite.Root 'expected_min')" `
        "--device=0" `
        "--warmup=0" `
        "--reps=1" |
        Tee-Object -FilePath (Join-Path $result "console.log")
    if ($LASTEXITCODE -ne 0) {
        throw "$($suite.Name) failed with exit code $LASTEXITCODE"
    }
}

Write-Output "[BLIN-OPTIM1D-RELAXED-FLATTEN-v0.6.0-CORRECTNESS-OK]"
