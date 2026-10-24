# Runs every GPU measurement reported in the paper, in a fixed order, from this
# repository. The BBBC038 distance maps and their marker statistics are derived
# beforehand on the CPU by datasets/bbbc038_distance/derive_bbbc038_distance_markers_v0_2.py. Run from an "x64 Native Tools" prompt (nvcc needs cl.exe) with
# SPARSE_SEAM_ROOT, OPENCV_ROOT and CUDA_PATH set, after the builds and the
# cuCIM environment of the README. The data\ folder of the workspace must
# already hold the datasets; nothing under data\ is written.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ErrorActionPreference = "Stop"

$results = Join-Path $ssRoot "results"
$logs = Join-Path $results "campaign_logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Invoke-Stage($name, [string[]]$scripts) {
    & (Join-Path $repo "tools\record_environment.ps1") -OutFile (Join-Path $logs "env_${name}_start.txt") | Out-Null
    foreach ($script in $scripts) {
        $log = Join-Path $logs ("{0}__{1}.log" -f $name, [IO.Path]::GetFileNameWithoutExtension($script))
        $started = Get-Date
        & pwsh.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File (Join-Path $repo $script) *> $log
        $code = $LASTEXITCODE
        $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
        Write-Output ("[{0}] {1} exit={2} minutes={3}" -f $name, $script, $code, $minutes)
        if ($code -ne 0) { throw "$script failed with exit code $code; see $log" }
    }
    & (Join-Path $repo "tools\record_environment.ps1") -OutFile (Join-Path $logs "env_${name}_end.txt") | Out-Null
}

# Stage 0: input identity. Hashes of every input image are recorded so that the
# campaign can be tied to the exact data it read.
$inputs = @(
    "data\DIV2K_valid_HR_official\DIV2K_valid_HR",
    "data\BSDS500_official_v0.1\images_v0.2",
    "data\BBBC038_stage1_train_derived_v0.3\derived_marker_watershed_v0.2\distance_maps",
    "data\synthetic_scaling_v0.1",
    "data\synthetic_scaling_alignment_v0.2"
)
$manifest = foreach ($folder in $inputs) {
    $full = Join-Path $ssRoot $folder
    foreach ($file in (Get-ChildItem -LiteralPath $full -File | Where-Object { $_.Extension -match '^\.(png|pgm|jpg|jpeg|bmp|tif|tiff)$' } | Sort-Object Name)) {
        [pscustomobject]@{
            folder = $folder
            file = $file.Name
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
}
$manifest | Export-Csv -LiteralPath (Join-Path $logs "input_manifest_sha256.csv") -NoTypeInformation
$manifest | Group-Object folder | ForEach-Object { Write-Output ("[inputs] {0} files={1}" -f $_.Name, $_.Count) }

Invoke-Stage "1_correctness" @(
    "harness\primary\build_run_gpu_common_v0_6.ps1",
    "harness\quantization_and_scaling\run_gpu_common_quant_v0_7_correctness.ps1",
    "baselines\npp_exact_repair\run_npp_labelmarkersuf_exact_repair_v0_3_correctness.ps1",
    "baselines\iwpp\run_iwpp_reconstruction_cuda_v0_1_correctness.ps1",
    "baselines\blin_dual_tree\run_blin_optim1d_relaxed_flatten_extrema_v0_6_0_correctness.ps1",
    "baselines\cucim\run_cucim_extrema_v0_1_correctness.ps1",
    "baselines\yacclab_gray\run_yacclab_gray_ccl_v0_2_correctness.ps1"
)
Invoke-Stage "2_div2k_and_new_baselines" @(
    "harness\primary\run_div2k_valid100.ps1",
    "baselines\npp_exact_repair\run_npp_labelmarkersuf_exact_repair_v0_3_benchmarks.ps1",
    "baselines\iwpp\run_iwpp_reconstruction_cuda_v0_1_div2k_30rep_v0_2.ps1",
    "baselines\blin_dual_tree\run_blin_optim1d_relaxed_flatten_extrema_v0_6_0_benchmarks.ps1",
    "baselines\cucim\run_cucim_extrema_v0_1_benchmarks.ps1",
    "baselines\yacclab_gray\run_yacclab_gray_ccl_v0_2_benchmarks.ps1"
)
Invoke-Stage "3_configuration" @("harness\configuration_study\run_config_study_v0_8.ps1")
Invoke-Stage "4_quantization" @(
    "harness\quantization_and_scaling\run_gpu_common_quant_v0_7_div2k.ps1",
    "baselines\yacclab_gray\run_yacclab_gray_ccl_v0_2_quantization.ps1"
)
Invoke-Stage "5_scaling" @(
    "datasets\synthetic_scaling\run_gpu_common_quant_v0_7_scaling_repeat_v0_2.ps1",
    "datasets\synthetic_alignment_control\run_gpu_common_quant_v0_7_alignment_repeat_v0_2.ps1",
    "baselines\yacclab_gray\run_yacclab_gray_ccl_v0_2_scaling.ps1",
    "baselines\yacclab_gray\run_yacclab_gray_ccl_v0_2_alignment.ps1"
)
Invoke-Stage "6_bsds500" @("harness\quantization_and_scaling\run_gpu_common_quant_v0_7_bsds500_v0_2.ps1")
Invoke-Stage "7_bbbc038" @("harness\quantization_and_scaling\run_gpu_common_quant_v0_7_bbbc038_distance_v0_3.ps1")
Invoke-Stage "8_seam_profile" @("harness\seam_sparsity_profiler\run_div2k_seam_profile_v0_2.ps1")
Invoke-Stage "9_ablation" @("harness\ablation_four_edge_local\run_ablation_four_edge_v0_6.ps1")
Write-Output "[ALL-CAMPAIGNS-OK]"
