# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = "$ssRoot"
$harness = Join-Path $repo "harness\primary"
$source = Join-Path $harness "gpu_common_v0_5.cu"
$executable = Join-Path $harness "gpu_common_v0_5.exe"
$nvcc = "$cudaRoot\bin\nvcc.exe"
$opencvRoot = "$ocvRoot"
$opencvInclude = Join-Path $opencvRoot "include"
$opencvLib = Join-Path $opencvRoot "x64\vc16\lib"
$opencvBin = Join-Path $opencvRoot "x64\vc16\bin"

& $nvcc `
    -O3 `
    -std=c++17 `
    -arch=sm_89 `
    -DSS_TILE=32 `
    -DSS_CLASSIFY_VARIANT=1 `
    -Xcompiler "/EHsc /O2" `
    "-I$opencvInclude" `
    "-L$opencvLib" `
    -lopencv_world4110 `
    $source `
    -o $executable
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

$env:PATH = "$opencvBin;$env:PATH"
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
        "results\gpu_common_v0.5_{0}_correctness_sog" -f
        $suite.Name)
    New-Item -ItemType Directory -Path $result -Force | Out-Null
    & $executable `
        (Join-Path $suite.Root $suite.Inputs) `
        "--output=$result" `
        "--expected-max=$(Join-Path $suite.Root 'expected_max')" `
        "--expected-min=$(Join-Path $suite.Root 'expected_min')" `
        "--device=0" `
        "--block=16" `
        "--warmup=1" `
        "--reps=1" `
        "--order=sog" |
        Tee-Object -FilePath (Join-Path $result "console.log")
    if ($LASTEXITCODE -ne 0) {
        throw "$($suite.Name) failed with exit code $LASTEXITCODE"
    }
}

Write-Output "[GPU-COMMON-v0.5-BUILD-RUN-OK]"
Write-Output "source_sha256=$((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash)"
Write-Output "executable_sha256=$((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash)"
