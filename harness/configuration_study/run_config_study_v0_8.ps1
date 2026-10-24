# SPARSE_SEAM_ROOT: workspace holding data\ and results\; sources and oracles are read from this repository.
# Set SPARSE_SEAM_ROOT before running; the value below is only a fallback.
if ($env:SPARSE_SEAM_ROOT) { $ssRoot = $env:SPARSE_SEAM_ROOT } else { $ssRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ($env:OPENCV_ROOT) { $ocvRoot = $env:OPENCV_ROOT } else { $ocvRoot = "C:\opencv\build" }
if ($env:CUDA_PATH) { $cudaRoot = $env:CUDA_PATH } else { $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" }
$ErrorActionPreference = "Stop"

# Builds the nine preregistered configurations, admits each on the 206-case
# oracle, then times them on DIV2K in nine rounds. Round r runs the nine
# binaries in the base order cyclically shifted by r-1, so every
# configuration occupies every execution position once. Each run uses five
# warm-ups and ten measured repetitions per image; the per-image result is
# the median over the nine round medians, and a configuration's throughput is
# total pixels divided by the sum of those per-image medians.
$here = $PSScriptRoot
$source = Join-Path $here "sparse_config_benchmark_v0_8.cu"
$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
$configs = @("16x256", "20x256", "24x256", "28x256", "32x256", "64x256",
             "32x128", "32x512", "32x1024")
$env:PATH = @(
    "$cudaRoot\bin"
    "$ocvRoot\x64\vc16\bin"
    $env:PATH
) -join ";"
$env:CUDA_MODULE_LOADING = "EAGER"
$dataset = Join-Path $ssRoot "data\DIV2K_valid_HR_official\DIV2K_valid_HR"
$resultRoot = Join-Path $ssRoot "results\config_study_v0_8"
if (Test-Path -LiteralPath $resultRoot) {
    throw "Refusing to overwrite existing result: $resultRoot"
}
New-Item -ItemType Directory -Path $resultRoot | Out-Null

function Get-ConfigName($config) {
    $tile, $threads = $config -split "x"
    return "tile${tile}_threads$threads"
}
function Get-Executable($config) {
    return Join-Path $here ("sparse_config_{0}.exe" -f (Get-ConfigName $config))
}

foreach ($config in $configs) {
    $tile, $threads = $config -split "x"
    & $nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/EHsc /O2" `
        "-DSS_TILE=$tile" "-DSS_CLASSIFY_VARIANT=1" "-DSS_CLASSIFY_THREADS=$threads" `
        "-I$(Join-Path $ocvRoot 'include')" "-L$(Join-Path $ocvRoot 'x64\vc16\lib')" `
        -lopencv_world4110 $source -o (Get-Executable $config)
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $config" }
}

$suites = @(
    @{ Name = "oracle34"; Root = Join-Path $repo "oracles\oracle_v0_1"; Inputs = "inputs_png" },
    @{ Name = "fuzz160"; Root = Join-Path $repo "oracles\oracle_v0_2_fuzz"; Inputs = "inputs" },
    @{ Name = "seam12"; Root = Join-Path $repo "oracles\oracle_v0_3_sparse_seam_adversarial"; Inputs = "inputs" }
)

foreach ($config in $configs) {
    $name = Get-ConfigName $config
    foreach ($suite in $suites) {
        & (Get-Executable $config) `
            (Join-Path $suite.Root $suite.Inputs) `
            (Join-Path $suite.Root "expected_max") `
            (Join-Path $suite.Root "expected_min") `
            "--output=$(Join-Path $resultRoot "admission\${name}_$($suite.Name)")" `
            "--round=0" "--config=$name" | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "Admission failed: $name $($suite.Name)" }
    }
}

# With --skip-expected the two expected-mask arguments are ignored; the image
# directory is passed in their place.
for ($round = 1; $round -le 9; ++$round) {
    for ($position = 0; $position -lt $configs.Count; ++$position) {
        $config = $configs[($position + $round - 1) % $configs.Count]
        $name = Get-ConfigName $config
        $output = Join-Path $resultRoot (
            "timing\round_{0:D2}\position_{1:D2}_{2}" -f $round, ($position + 1), $name)
        & (Get-Executable $config) $dataset $dataset $dataset `
            "--output=$output" "--device=0" "--warmup=5" "--reps=10" `
            "--round=$round" "--config=$name" "--skip-expected" | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "Timing failed: round $round $name" }
    }
}

$rows = foreach ($summary in Get-ChildItem -Path (Join-Path $resultRoot "timing") -Recurse -Filter "summary.csv") {
    Import-Csv -LiteralPath $summary.FullName
}
$aggregate = foreach ($group in ($rows | Group-Object config)) {
    $pixels = 0.0
    $seconds = 0.0
    $auxiliary = 0.0
    foreach ($image in ($group.Group | Group-Object image)) {
        if ($image.Count -ne 9) { throw "Expected nine rounds for $($group.Name) $($image.Name)" }
        $medians = @($image.Group | ForEach-Object { [double]$_.median_compute_ms } | Sort-Object)
        $first = $image.Group[0]
        $pixels += [double]$first.width * [double]$first.height
        $seconds += $medians[4] / 1000.0
        $auxiliary += [double]$first.auxiliary_bytes
    }
    [pscustomobject]@{
        config = $group.Name
        images = @($group.Group | Group-Object image).Count
        aggregate_mpix_s = [math]::Round($pixels / 1e6 / $seconds, 3)
        auxiliary_Bpp = [math]::Round($auxiliary / $pixels, 6)
    }
}
$aggregate | Sort-Object aggregate_mpix_s -Descending |
    Export-Csv -LiteralPath (Join-Path $resultRoot "config_study_summary.csv") -NoTypeInformation
$aggregate | Sort-Object aggregate_mpix_s -Descending | Format-Table -AutoSize | Out-String | Write-Output
Write-Output "[CONFIG-STUDY-v0.8-OK]"
