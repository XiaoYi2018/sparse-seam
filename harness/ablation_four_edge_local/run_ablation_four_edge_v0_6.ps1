# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

# Ablation of the tile-local labeling stage: the Sparse-Seam v0.9 core, whose
# tile-local stage unites every equal pixel with its four predecessors from
# singleton parents, measured with the same quantization harness and protocol
# as the v1.0 core (harness\quantization_and_scaling). Pair these results with
# the v1.0 runs of run_gpu_common_quant_v0_7_div2k.ps1, *_bsds500_v0_2.ps1 and
# *_bbbc038_distance_v0_3.ps1 from the same session.
$executable = Join-Path $repo "harness\ablation_four_edge_local\gpu_common_quant_v0_6.exe"
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"
$runs = @(
    @{ Name = "DIV2K_q256"; Levels = 256; Inputs = "data\DIV2K_valid_HR_official\DIV2K_valid_HR" },
    @{ Name = "DIV2K_q32"; Levels = 32; Inputs = "data\DIV2K_valid_HR_official\DIV2K_valid_HR" },
    @{ Name = "DIV2K_q8"; Levels = 8; Inputs = "data\DIV2K_valid_HR_official\DIV2K_valid_HR" },
    @{ Name = "DIV2K_q2"; Levels = 2; Inputs = "data\DIV2K_valid_HR_official\DIV2K_valid_HR" },
    @{ Name = "BSDS500_v0.2"; Levels = 256; Inputs = "data\BSDS500_official_v0.1\images_v0.2" },
    @{ Name = "BBBC038_distance_v0.3"; Levels = 256; Inputs = "data\BBBC038_stage1_train_derived_v0.3\derived_marker_watershed_v0.2\distance_maps" }
)
foreach ($run in $runs) {
    foreach ($order in @("gos", "sog")) {
        $result = Join-Path $ssRoot ("results\ablation_four_edge_v0.6_{0}_w5_r30_{1}" -f $run.Name, $order)
        if (Test-Path -LiteralPath $result) {
            throw "Refusing to overwrite existing result: $result"
        }
        New-Item -ItemType Directory -Path $result | Out-Null
        & $executable `
            (Join-Path $ssRoot $run.Inputs) `
            "--output=$result" `
            "--device=0" `
            "--block=16" `
            "--warmup=5" `
            "--reps=30" `
            "--order=$order" `
            "--levels=$($run.Levels)" |
            Tee-Object -FilePath (Join-Path $result "console.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$($run.Name) $order failed with exit code $LASTEXITCODE"
        }
    }
}
Write-Output "[ABLATION-FOUR-EDGE-v0.6-OK]"
