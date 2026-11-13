# 溯忆 · 一键测试
#
# 一条命令跑完所有该跑的检查。之前是「这个测一下、那个测一下」，
# 容易漏——漏掉的测试等于没有测试。
#
# 用法：
#   .\run_tests.ps1              # 全部
#   .\run_tests.ps1 -Fast        # 跳过需要服务的（离线逻辑测试）
#
# 退出码：0 = 全过，1 = 有失败（方便以后接 CI）

param(
    [switch]$Fast
)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspace = Split-Path -Parent $root
$py = Join-Path $workspace '_inspect\mnemosyne\service\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    # 换机器/换目录时的退路：用 PATH 里的 python
    $py = 'python'
}
if ($env:HF_ENDPOINT -eq $null) { $env:HF_ENDPOINT = 'https://hf-mirror.com' }

$failed = @()
$passed = @()

function Run-Check($name, $cmd, $argList, $cwd) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    Push-Location $cwd
    try {
        & $cmd @argList
        if ($LASTEXITCODE -eq 0) { $script:passed += $name } else { $script:failed += $name }
    } catch {
        Write-Host "  异常: $_" -ForegroundColor Red
        $script:failed += $name
    } finally {
        Pop-Location
    }
}

Write-Host "溯忆 · 一键测试" -ForegroundColor Yellow
Write-Host "python: $py"

# 1) 语法检查（所有 py 都要能 import）
Run-Check 'py_compile' $py @('-m', 'py_compile',
    (Join-Path $root 'persona_memory\service.py'),
    (Join-Path $root 'persona_memory\api.py'),
    (Join-Path $root 'persona_memory\ui.py'),
    (Join-Path $root 'persona_memory\main.py'),
    (Join-Path $root 'persona_memory\embedding.py'),
    (Join-Path $root 'persona_memory\interest_harvest.py'),
    (Join-Path $root 'persona_memory\correction_harvest.py'),
    (Join-Path $root 'watchdog.py'),
    (Join-Path $root 'dedupe.py'),
    (Join-Path $root 'clean_noise.py'),
    (Join-Path $root 'backfill_links.py'),
    (Join-Path $root 'rebuild_index.py')
) $root

# 2) 离线逻辑测试（不需要服务）
Run-Check 'test_smoke'      $py @(Join-Path $root 'test_smoke.py')   $root
Run-Check 'test_harvest'    $py @(Join-Path $root 'test_harvest.py') $root
Run-Check 'test_gaps'       $py @(Join-Path $root 'test_gaps.py')    $root
Run-Check 'test_emotion'    $py @(Join-Path $root 'test_emotion.py') $root
Run-Check 'test_persist'    $py @(Join-Path $root 'test_persist.py') $root
Run-Check 'test_garbled'    $py @(Join-Path $root 'test_garbled.py') $root
Run-Check 'test_links'      $py @(Join-Path $root 'test_links.py')   $root

if (-not $Fast) {
    # 3) 需要服务的（会写真实记忆库，跑完可用 cleanup_junk.py 清）
    Run-Check 'test_api' $py @(Join-Path $root 'test_api.py') $root

    # 4) 插件测试（会连真实服务做一次连通验证）
    $plugin = Join-Path (Split-Path -Parent $root) 'dsh-suyi'
    if (Test-Path $plugin) {
        Run-Check 'test-plugin' 'node' @(Join-Path $plugin 'test\test-plugin.mjs') $plugin
    }

    # 5) 记忆体检：这是「报告」不是「断言」——有警告只提示，不算测试失败
    Write-Host ""
    Write-Host "=== watchdog（体检，不做断言） ===" -ForegroundColor Cyan
    Push-Location $root
    & $py (Join-Path $root 'watchdog.py') --fix
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
if ($failed.Count -eq 0) {
    Write-Host "RESULT: ALL PASS ($($passed.Count) 项)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "RESULT: $($failed.Count) 项失败" -ForegroundColor Red
    foreach ($f in $failed) { Write-Host "  - $f" -ForegroundColor Red }
    Write-Host "通过: $($passed -join ', ')"
    exit 1
}
