py -m venv .venv

$filePath = "$PSScriptRoot\.venv\Scripts\Activate.ps1"
if(-not (Test-Path -Path $filePath)) {
    throw "Run setup again"
}

. "$PSScriptRoot\.venv\Scripts\Activate.ps1"

if(-not (Test-Path env:VIRTUAL_ENV)) {
    throw "Run setup again"
}

python -m pip install --upgrade pip
pip install -r requirements.txt