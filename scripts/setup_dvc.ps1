param(
    [string]$DagsHubUsername = $env:DAGSHUB_USERNAME,
    [string]$DagsHubToken = $(if ($env:DAGSHUB_TOKEN) { $env:DAGSHUB_TOKEN } else { $env:DAGSHUB_USER_TOKEN }),
    [string]$RemoteName = "dagshub"
)

$ErrorActionPreference = "Stop"

if (-not $DagsHubUsername -or -not $DagsHubToken) {
    Write-Error "Missing credentials. Provide -DagsHubUsername and -DagsHubToken or set DAGSHUB_USERNAME and DAGSHUB_TOKEN (or DAGSHUB_USER_TOKEN)."
}

Write-Host "Configuring DVC remote '$RemoteName' with local credentials..."

dvc remote modify $RemoteName --local access_key_id "$DagsHubUsername"
dvc remote modify $RemoteName --local secret_access_key "$DagsHubToken"

Write-Host "Validating DVC remote by listing config..."
dvc remote list

Write-Host "Done. You can now run:"
Write-Host "  dvc pull"
Write-Host "  dvc repro"
Write-Host "  dvc push"
