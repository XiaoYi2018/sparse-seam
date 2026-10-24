param(
    [Parameter(Mandatory = $true)]
    [string]$SourceTarball,
    [string]$Python = "python"
)
# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

# cuCIM is used unmodified from its v26.10.00 release source (Apache-2.0); its
# scikit-image layer is pure Python on CuPy, so it runs natively on Windows
# from the release source tree. Download the archive from
# https://github.com/rapidsai/cucim/archive/refs/tags/v26.10.00.tar.gz
$expected = "53F95163F450B6E355D11E2870038DB51D6CCBD23214B3C1CDAB9A30F8546063"
if ((Get-FileHash -LiteralPath $SourceTarball -Algorithm SHA256).Hash -ne $expected) {
    throw "cuCIM source archive hash mismatch"
}
$thirdParty = Join-Path $PSScriptRoot "third_party"
$venv = Join-Path $PSScriptRoot "venv"
foreach ($path in @($thirdParty, $venv)) {
    if (Test-Path -LiteralPath $path) { throw "Already exists: $path" }
}
New-Item -ItemType Directory -Path $thirdParty | Out-Null
# src/cucim/VERSION is a symbolic link to the archive-root VERSION file, which
# is therefore extracted as well. The Windows tar is named explicitly so that
# another tar earlier on PATH cannot misread the drive letter as a host.
& (Join-Path $env:SystemRoot "System32\tar.exe") -xzf $SourceTarball -C $thirdParty `
    "cucim-26.10.00/python/cucim/src/cucim" `
    "cucim-26.10.00/VERSION" `
    "cucim-26.10.00/LICENSE"
if ($LASTEXITCODE -ne 0) { throw "tar failed with exit code $LASTEXITCODE" }
& $Python -m venv $venv
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --quiet -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
$env:PYTHONPATH = Join-Path $thirdParty "cucim-26.10.00\python\cucim\src"
& $py -c "import cupy, cucim; from cucim.skimage.morphology import local_maxima; print('cupy', cupy.__version__, 'cucim', cucim.__version__)"
if ($LASTEXITCODE -ne 0) { throw "cuCIM import check failed" }
Write-Output "[CUCIM-ENV-OK]"
