param(
    [string]$Output
)
# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

# Writes the aligned and half-tile-shifted block suites (12 images) and their
# manifest. The generator is deterministic; an existing output is never
# overwritten.
$executable = Join-Path $repo "datasets\synthetic_alignment_control\generate_scaling_suite_v0_2.exe"
if (-not $Output) {
    $Output = Join-Path $ssRoot "data\synthetic_scaling_alignment_v0.2"
}
if (Test-Path -LiteralPath $Output) {
    throw "Refusing to overwrite existing output: $Output"
}
& $executable $Output
if ($LASTEXITCODE -ne 0) {
    throw "generator failed with exit code $LASTEXITCODE"
}
Get-ChildItem -LiteralPath $Output -File |
    Sort-Object Name |
    ForEach-Object { "{0}  {1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash, $_.Name }
Write-Output "[ALIGNMENT-SUITE-v0.2-OK]"
