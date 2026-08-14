[CmdletBinding()]
param(
    [string]$RobotHost = "10.200.6.146",
    [string]$UserName = "admin",
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$HostKey,
    [System.Management.Automation.PSCredential]$Credential,
    [Security.SecureString]$SudoPassword,
    [string]$RemoteMujocoRoot = "",
    [string]$Plink = "C:\Program Files\PuTTY\plink.exe",
    [string]$Pscp = "C:\Program Files\PuTTY\pscp.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
foreach ($tool in @($Plink, $Pscp)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "PuTTY tool not found: $tool" }
}
if (-not $Credential) { $Credential = Get-Credential -UserName $UserName -Message "SSH credentials for $RobotHost" }
if (-not $Credential) { throw "SSH authentication was cancelled." }
$UserName = $Credential.UserName
if (-not $SudoPassword) { $SudoPassword = $Credential.Password }
if ($RemoteMujocoRoot -and $RemoteMujocoRoot -notmatch '^/[A-Za-z0-9._/+@-]+(?:/[A-Za-z0-9._+@-]+)*$') {
    throw "RemoteMujocoRoot must be an absolute Linux path without shell metacharacters"
}

function ConvertTo-PlainText([Security.SecureString]$Value) { [Net.NetworkCredential]::new("", $Value).Password }
function New-PasswordFile([Security.SecureString]$Value) {
    $path = Join-Path ([IO.Path]::GetTempPath()) "kengo-retarget-auth-$([guid]::NewGuid().ToString('N')).txt"
    [IO.File]::WriteAllText($path, (ConvertTo-PlainText $Value), [Text.UTF8Encoding]::new($false))
    & icacls $path /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
    if ($LASTEXITCODE -ne 0) { Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue; throw "failed to protect password file" }
    $path
}

$payload = @(
    ".gitattributes", ".gitignore", "README.md", "requirements-realtime.txt",
    "config/robot/kengo.yaml",
    "asset/robot/kengo_description/README.md",
    "asset/robot/kengo_description/mjcf/kengo.xml",
    "cpp/CMakeLists.txt", "cpp/package.xml", "cpp/composed_target/CMakeLists.txt", "cpp/composed_target/package.xml",
    "cpp/include/kengo_fullbody/command_composer.hpp", "cpp/include/kengo_fullbody/composed_target.hpp", "cpp/include/kengo_fullbody/retarget_core.hpp",
    "cpp/src/fullbody_command_bridge_node.cpp", "cpp/src/fullbody_command_composer.cpp", "cpp/src/fullbody_command_composer_test.cpp",
    "cpp/src/fullbody_composed_target.cpp", "cpp/src/fullbody_composed_target_node.cpp", "cpp/src/fullbody_composed_target_test.cpp",
    "cpp/src/fullbody_graft_test.cpp", "cpp/src/fullbody_retarget_core.cpp", "cpp/src/fullbody_retarget_node.cpp",
    "deployment/build_realtime_fullbody_cpp.sh", "deployment/install.sh",
    "deployment/run_realtime_fullbody.sh", "deployment/run_fullbody_composed_target.sh",
    "deployment/kengo-fullbody-retarget.service", "deployment/kengo-fullbody-composed-target.service",
    "deployment/FULLBODY_CPP.md", "deployment/FULLBODY_COMPOSED_TARGET.md"
)
$meshes = @(Get-ChildItem -LiteralPath (Join-Path $repo "asset\robot\kengo_description\meshes") -File -Filter *.STL | Sort-Object Name)
if ($meshes.Count -ne 27) { throw "expected exactly 27 Kengo STL files, found $($meshes.Count)" }
$payload += $meshes | ForEach-Object { "asset/robot/kengo_description/meshes/$($_.Name)" }
foreach ($name in $payload) {
    $path = Join-Path $repo ($name -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "payload file missing: $name" }
    if ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "payload file must not be a link: $name" }
}

$nonce = [guid]::NewGuid().ToString("N")
$archive = Join-Path ([IO.Path]::GetTempPath()) "kengo-fullbody-$nonce.tar.gz"
$remoteDir = "/tmp/kengo-fullbody-upload-$nonce"
$passwordFile = New-PasswordFile $Credential.Password
$sudoPlain = ConvertTo-PlainText $SudoPassword
$ssh = @("-batch", "-hostkey", $HostKey, "-pwfile", $passwordFile)
$installSucceeded = $false
try {
    & tar -czf $archive -C $repo -- @payload
    if ($LASTEXITCODE -ne 0) { throw "failed to build exact retarget payload" }
    & $Plink @ssh "$UserName@$RobotHost" "install -d -m 700 '$remoteDir'"
    if ($LASTEXITCODE -ne 0) { throw "failed to create private remote upload directory" }
    & $Pscp @ssh $archive "$UserName@${RobotHost}:$remoteDir/payload.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "failed to upload retarget payload" }
    $remoteCommand = @'
set -Eeuo pipefail
upload='__UPLOAD__'
[[ "$upload" =~ ^/tmp/kengo-fullbody-upload-[0-9a-f]{32}$ ]]
[[ -d "$upload" && ! -L "$upload" && -O "$upload" ]]
[[ "$(stat -c '%a' -- "$upload")" == 700 ]]
mkdir "$upload/source"
tar -xzf "$upload/payload.tar.gz" -C "$upload/source" --no-same-owner --no-same-permissions
__MUJOCO__sudo -S -p '' -- bash "$upload/source/deployment/install.sh"
'@.Replace('__UPLOAD__', $remoteDir)
    $mujocoPrefix = if ($RemoteMujocoRoot) { "KENGO_MUJOCO_ROOT='$RemoteMujocoRoot' " } else { "" }
    $remoteCommand = $remoteCommand.Replace('__MUJOCO__', $mujocoPrefix)
    $sudoPlain | & $Plink @ssh "$UserName@$RobotHost" $remoteCommand
    if ($LASTEXITCODE -ne 0) { throw "Kengo full-body installation failed" }
    $installSucceeded = $true
}
finally {
    $sudoPlain = $null
    $cleanupSudoPlain = ConvertTo-PlainText $SudoPassword
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cleanupCommand = @'
set -Eeuo pipefail
upload='__UPLOAD__'
[[ "$upload" =~ ^/tmp/kengo-fullbody-upload-[0-9a-f]{32}$ ]]
[[ ! -e "$upload" && ! -L "$upload" ]] && exit 0
[[ -d "$upload" && ! -L "$upload" && -O "$upload" ]]
[[ "$(stat -c '%a' -- "$upload")" == 700 ]]
sudo -S -p '' -- rm -rf -- "$upload"
[[ ! -e "$upload" && ! -L "$upload" ]]
'@.Replace('__UPLOAD__', $remoteDir)
        $cleanupSudoPlain | & $Plink @ssh "$UserName@$RobotHost" $cleanupCommand | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Warning "remote cleanup failed: $remoteDir" }
    }
    catch { Write-Warning "remote cleanup failed: $remoteDir ($($_.Exception.Message))" }
    finally {
        $cleanupSudoPlain = $null
        $ErrorActionPreference = $savedPreference
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $passwordFile -Force -ErrorAction SilentlyContinue
        if ($installSucceeded) { $global:LASTEXITCODE = 0 }
    }
}
