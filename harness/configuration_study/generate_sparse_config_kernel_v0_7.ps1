param(
    [Parameter(Mandatory = $true)]
    [string]$CanonicalSource,

    [Parameter(Mandatory = $true)]
    [string]$GeneratedSource,

    [Parameter(Mandatory = $true)]
    [string]$AuditPath
)

$ErrorActionPreference = "Stop"

$expectedCanonicalHash = "29F74737DCD0A372B15F4AD3BF4A9CDBB9734B20A67A37A7EC37A4A8A86BA2C7"

function Write-NewUtf8File {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath,

        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $encoding = [System.Text.UTF8Encoding]::new($false)
    $bytes = $encoding.GetBytes($Text)
    $stream = [System.IO.FileStream]::new(
        $LiteralPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None)
    try {
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    } finally {
        $stream.Dispose()
    }
}

$CanonicalSource = [System.IO.Path]::GetFullPath($CanonicalSource)
$GeneratedSource = [System.IO.Path]::GetFullPath($GeneratedSource)
$AuditPath = [System.IO.Path]::GetFullPath($AuditPath)
$normalizedPaths = @($CanonicalSource, $GeneratedSource, $AuditPath)
for ($left = 0; $left -lt $normalizedPaths.Count; ++$left) {
    for ($right = $left + 1; $right -lt $normalizedPaths.Count; ++$right) {
        if ([string]::Equals(
                $normalizedPaths[$left],
                $normalizedPaths[$right],
                [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Canonical, generated, and audit paths must be distinct"
        }
    }
}

foreach ($newPath in @($GeneratedSource, $AuditPath)) {
    if (Test-Path -LiteralPath $newPath) {
        throw "Refusing to overwrite existing generated artifact: $newPath"
    }
}
if (-not (Test-Path -LiteralPath $CanonicalSource)) {
    throw "Missing canonical source: $CanonicalSource"
}
$canonicalHash = (Get-FileHash -LiteralPath $CanonicalSource -Algorithm SHA256).Hash
if ($canonicalHash -ne $expectedCanonicalHash) {
    throw "Canonical source hash mismatch: $canonicalHash"
}

$content = Get-Content -LiteralPath $CanonicalSource -Raw

$assertOld = 'static_assert(TILE == 32 || TILE == 64, "SS_TILE must be 32 or 64");'
$assertNew = @'
static_assert(
    TILE == 16 || TILE == 20 || TILE == 24 ||
    TILE == 28 || TILE == 32 || TILE == 64,
    "SS_TILE is outside the preregistered configuration set");
'@

$threadsOld = "    constexpr int threads = 256;"
$threadsNew = @'
#ifndef SS_CLASSIFY_THREADS
#define SS_CLASSIFY_THREADS 256
#endif
    constexpr int classify_threads = SS_CLASSIFY_THREADS;
    constexpr int threads = 256;
'@

$launchOld = "classify_tiles_and_emit_fragments<<<tile_grid, threads>>>("
$launchNew = "classify_tiles_and_emit_fragments<<<tile_grid, classify_threads>>>("

$changes = @(
    [pscustomobject]@{ Name="tile_assertion"; Old=$assertOld; New=$assertNew },
    [pscustomobject]@{ Name="classification_thread_macro"; Old=$threadsOld; New=$threadsNew },
    [pscustomobject]@{ Name="classification_launch"; Old=$launchOld; New=$launchNew }
)

$generated = $content
foreach ($change in $changes) {
    $first = $generated.IndexOf($change.Old)
    $last = $generated.LastIndexOf($change.Old)
    if ($first -lt 0 -or $first -ne $last) {
        throw "Expected exactly one occurrence for $($change.Name)"
    }
    $generated = $generated.Replace($change.Old, $change.New)
}

foreach ($change in $changes) {
    # The classification-thread replacement deliberately retains the canonical
    # "constexpr int threads = 256;" line for the other four launches.  Only
    # require the old text to disappear when the audited replacement does not
    # itself contain that old text.
    if (-not $change.New.Contains($change.Old) -and
        $generated.Contains($change.Old)) {
        throw "Old text remains after replacement: $($change.Name)"
    }
    $first = $generated.IndexOf($change.New)
    $last = $generated.LastIndexOf($change.New)
    if ($first -lt 0 -or $first -ne $last) {
        throw "Expected exactly one generated occurrence: $($change.Name)"
    }
}

Write-NewUtf8File -LiteralPath $GeneratedSource -Text $generated

$generatedHash = (Get-FileHash -LiteralPath $GeneratedSource -Algorithm SHA256).Hash
$audit = @(
    "generator=generate_sparse_config_kernel_v0.7"
    "canonical_path=$CanonicalSource"
    "canonical_sha256=$canonicalHash"
    "generated_path=$GeneratedSource"
    "generated_sha256=$generatedHash"
    "allowed_change_count=3"
    "allowed_change_1=tile compile-time assertion only"
    "allowed_change_2=classification thread macro/declaration only"
    "allowed_change_3=classification kernel launch width only"
    "other_kernel_launch_width=256"
    "device_algorithm_semantics_changed=false"
)
$auditText = ($audit -join [System.Environment]::NewLine) +
    [System.Environment]::NewLine
Write-NewUtf8File -LiteralPath $AuditPath -Text $auditText

$canonicalHashAfter =
    (Get-FileHash -LiteralPath $CanonicalSource -Algorithm SHA256).Hash
$generatedHashAfter =
    (Get-FileHash -LiteralPath $GeneratedSource -Algorithm SHA256).Hash
if ($canonicalHashAfter -ne $canonicalHash -or
    $generatedHashAfter -ne $generatedHash) {
    throw "Canonical or generated source changed before generator completion"
}

Write-Output "[SPARSE-CONFIG-KERNEL-GENERATOR-v0.7-OK]"
Write-Output "canonical_sha256=$canonicalHash"
Write-Output "generated_sha256=$generatedHash"
