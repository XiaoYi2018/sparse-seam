# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

$root = "$ssRoot"
$harness = Join-Path $repo "baselines\npp_exact_repair"
$source = Join-Path $harness "npp_labelmarkersuf_exact_repair_v0_3.cu"
$executable = Join-Path $harness "npp_labelmarkersuf_exact_repair_v0_3.exe"
$cudaRoot = "$cudaRoot"
$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
$opencvRoot = "$ocvRoot"

& $nvcc `
    -O3 `
    -std=c++17 `
    -arch=sm_89 `
    -Xcompiler "/EHsc /O2" `
    "-I$harness" `
    "-I$(Join-Path $cudaRoot 'include')" `
    "-I$(Join-Path $opencvRoot 'include')" `
    "-L$(Join-Path $cudaRoot 'lib\x64')" `
    "-L$(Join-Path $opencvRoot 'x64\vc16\lib')" `
    -lnppif `
    -lnppc `
    -lopencv_world4110 `
    $source `
    -o $executable
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Output "[NPP-UF-REPAIR-v0.3-BUILD-OK]"
Write-Output "source_sha256=$((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash)"
Write-Output "executable_sha256=$((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash)"
