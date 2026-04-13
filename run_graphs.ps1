Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Rscript {
    $command = Get-Command Rscript -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "C:\Program Files (x86)\R\R-*\bin\Rscript.exe",
        "C:\Program Files\R\R-*\bin\Rscript.exe",
        "C:\Program Files (x86)\R\R-*\bin\x64\Rscript.exe",
        "C:\Program Files\R\R-*\bin\x64\Rscript.exe"
    )

    foreach ($pattern in $candidates) {
        $match = Get-ChildItem $pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    throw "Rscript.exe was not found. Install R or add Rscript to PATH, then rerun this script."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$rscript = Resolve-Rscript
$analysisScript = Join-Path $repoRoot "analysis\explanatory_graphs_r\make_graphs.R"

if (-not (Test-Path $analysisScript)) {
    throw "Analysis script not found at $analysisScript"
}

Write-Host "Using Rscript at $rscript"
& $rscript $analysisScript
