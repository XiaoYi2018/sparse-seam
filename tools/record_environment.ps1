param(
    [Parameter(Mandatory = $true)]
    [string]$OutFile
)
$ErrorActionPreference = "Stop"
# Records the state that changes measured throughput: driver, clocks, power
# limit, power source and plan, and other processes holding the GPU.
$lines = New-Object System.Collections.Generic.List[string]
$os = Get-CimInstance Win32_OperatingSystem
$lines.Add("timestamp=$(Get-Date -Format o)")
$lines.Add("os=$($os.Caption) build $($os.BuildNumber)")
$lines.Add("power_plan=$((powercfg /getactivescheme) -replace '^.*GUID:\s*', '')")
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery) { $lines.Add("battery_status=$($battery.BatteryStatus) (2 = on AC power)") }
$query = "name,driver_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu"
$lines.Add("gpu=$(nvidia-smi --query-gpu=$query --format=csv,noheader)")
$lines.Add("gpu_fields=$query")
$lines.Add("nvcc=$((& (Join-Path $env:CUDA_PATH 'bin\nvcc.exe') --version | Select-String 'release').ToString().Trim())")
foreach ($line in (nvidia-smi | Select-String '^\|\s+\d+\s+N/A\s+N/A\s+\d+\s+')) {
    $lines.Add("gpu_process=$($line.ToString().Trim())")
}
$lines | Set-Content -LiteralPath $OutFile -Encoding ascii
$lines | Write-Output
