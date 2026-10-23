# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$profilerDir = "$repo\harness\seam_sparsity_profiler"
$source = Join-Path $profilerDir "gpu_seam_sparsity_profiler_v0_1.cu"
$executable = Join-Path $profilerDir "gpu_seam_sparsity_profiler_v0_1.exe"
$nvcc = "$cudaRoot\bin\nvcc.exe"
$opencvRoot = "$ocvRoot"
$opencvInclude = Join-Path $opencvRoot "include"
$opencvLib = Join-Path $opencvRoot "x64\vc16\lib"

if (-not (Test-Path -LiteralPath $source)) {
    throw "Missing source: $source"
}

& $nvcc `
    -O3 `
    -std=c++17 `
    -arch=sm_89 `
    -Xcompiler "/EHsc /O2" `
    "-I$opencvInclude" `
    "-L$opencvLib" `
    -lopencv_world4110 `
    $source `
    -o $executable

if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
$executableHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash
Write-Output "[GPU-SEAM-PROFILER-BUILD-OK]"
Write-Output "source_sha256=$sourceHash"
Write-Output "executable_sha256=$executableHash"
