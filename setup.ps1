# setup.ps1

$venvPath = "$PSScriptRoot\.venv"

if (-not (Test-Path -Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -3.11 -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists, skipping creation." -ForegroundColor Yellow
}

$activationScript = "$venvPath\Scripts\Activate.ps1"
if (-not (Test-Path -Path $activationScript)) {
    throw "Failed to create virtual environment. Please delete the .venv folder and try again."
}

. $activationScript

if (-not (Test-Path env:VIRTUAL_ENV)) {
    throw "Activation failed. Your PowerShell Execution Policies might be blocking scripts."
}

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r "$PSScriptRoot\requirements.txt"

Write-Host "Setup complete! Environment is active." -ForegroundColor Green