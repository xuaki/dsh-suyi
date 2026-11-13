# 溯忆 · Anamnesis 部署指南

> 从零部署一套**干净的**溯忆系统。不包含任何既有记忆、人格状态或个人数据。
> 适用：Windows 10/11 + DSH 0.2.x + Node 20+ + Python 3.11+

---

## 0. 你会得到什么

两样东西：

1. **`dsh-suyi`** —— DSH 插件（JS）。装进 profile 后，DSH 每次启动都会自动拉起下面那个服务。
2. **`persona-memory`** —— 本地 Python 服务（FastAPI，默认端口 8772）。

外加它的依赖服务：

3. **`mnemosyne`** —— 证据链记忆服务（FastAPI，默认端口 8765），SQLite 为权威源。
4. **Milvus** —— 向量索引（可选但强烈建议）。跑在 WSL2 里，端口 19530。

**第一次启动会是空的**：没有记忆、人格是出厂值、执念池空、兴趣空。它不是"删了数据"，是"从没记过"。

---

## 1. 前置条件

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| DSH | >= 0.2.0 | 桌面版或 CLI |
| Node.js | >= 20（推荐 22+） | 插件运行 |
| Python | 3.11 / 3.12 | 服务运行（建议独立 venv） |
| WSL2 + Ubuntu | 22.04+ | 仅当要跑 Milvus |
| Milvus | 3.0+（或 milvus-lite） | 向量索引，可选 |

> **没有 Milvus 也能跑**：系统会降级到关键词检索（`vector_up: false`），人格/记忆/执念全部正常，只是语义召回弱一些。

---

## 2. 目录规划

推荐把两个包放在同一个工作目录下，插件默认按这个布局推断路径：

```
<workspace>/
├── dsh-suyi/                 # 插件源码（JS）
│   ├── src/
│   │   ├── index.js
│   │   ├── client.js
│   │   ├── bootstrap.js
│   │   ├── tool-spec.js
│   │   ├── persona.js
│   │   └── fast-path.js
│   ├── persona.json          # 人格三段（身份/性格/思考）★出厂文件
│   ├── data/                 # 运行日志
│   ├── package.json
│   ├── dsh.plugin.json
│   └── cordis.patch.yml
│
├── persona-memory/           # 本地服务（Python）
│   ├── persona_memory/
│   │   ├── main.py           # 入口（--port）
│   │   ├── api.py            # FastAPI 路由
│   │   ├── service.py        # 记忆编排
│   │   ├── persona.py        # 人格三层
│   │   ├── molecules.py      # 内分泌
│   │   ├── admission.py      # 准入判定
│   │   ├── gaps.py           # 缺口池
│   │   ├── interests.py      # 兴趣画像
│   │   ├── presence.py       # 在场感
│   │   ├── usage.py          # 使用账本
│   │   ├── clusters.py       # 记忆分簇
│   │   ├── milvus_store.py   # 向量层
│   │   ├── mnemosyne_client.py
│   │   └── ui.py             # 管理看板
│   ├── data/                 # ★运行状态（首次启动自动创建）
│   │   ├── genome.json       # 基因组（出厂文件，可随包分发）
│   │   ├── persona_state.json# 人格状态（用户数据）
│   │   ├── gaps.json         # 执念池（用户数据）
│   │   ├── interests.json    # 兴趣画像（用户数据）
│   │   ├── presence.json     # 在场感（用户数据）
│   │   └── usage.json        # 使用账本（用户数据）
│   ├── watchdog.py           # 体检
│   ├── dedupe.py             # 去重
│   ├── rebuild_index.py      # 重建向量索引
│   └── requirements.txt
│
└── _inspect/
    ├── mnemosyne/service/    # mnemosyne 服务源码 + .venv
    └── mnemosyne-data/       # mnemosyne 数据（SQLite）★用户数据
        └── data/memory.db
```

> **★ 标记**：`genome.json` 与 `persona.json` 是"出厂设定"，可以随包分发；
> 其余 `data/` 下的 json 与 `mnemosyne-data/` 全是**用户数据**，部署时必须为空。

---

## 3. 安装

### 3.1 准备 Python 环境

```powershell
cd <workspace>\persona-memory
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 至少包含：

```
fastapi
uvicorn[standard]
pydantic
pymilvus
fastembed
numpy
```

### 3.2 准备 mnemosyne

按 mnemosyne 仓库说明安装，得到一个可运行的 `mnemosyne` 模块，以及它的 venv：

```
<workspace>\_inspect\mnemosyne\service\.venv\Scripts\python.exe
```

（插件默认从这里找 Python 解释器。改路径见 §4.3。）

### 3.3 装 Milvus（可选）

任选一种：

- **Docker / WSL**：`milvus run standalone`，监听 `19530`
- **milvus-lite**：`from pymilvus import MilvusClient; MilvusClient("milvus_lite.db")`（走本地文件）

> 用 WSL 时注意：Windows→WSL 的 localhost 转发经常在 WSL 重启后失效。
> 插件已内置兜底——启动时自动 `wsl hostname -I` 取 IP，通过 `MILVUS_URI` 传给服务。
> 也可以手动指定环境变量 `MILVUS_URI=http://<ip>:19530`。

### 3.4 装插件进 DSH profile

编辑 `<DSH_HOME>/profiles/<profile>/package.json`：

```json
{
  "dependencies": {
    "dsh-suyi": "link:<workspace>/dsh-suyi"
  },
  "dsh": {
    "profile": {
      "bundles": [
        "...",
        "dsh-suyi"
      ]
    }
  }
}
```

然后在 `node_modules` 下建一个目录符号链接（Windows）：

```powershell
New-Item -ItemType Junction `
  -Path "<DSH_HOME>\profiles\<profile>\node_modules\dsh-suyi" `
  -Target "<workspace>\dsh-suyi"
```

（如果 pnpm 可用，直接 `pnpm install` 也行。）

---

## 4. 配置

### 4.1 人格（`dsh-suyi/persona.json`）

三段，各自一个开关：

```json
{
  "identity":    { "enabled": true, "text": "……" },
  "personality": { "enabled": true, "text": "……" },
  "thinking":    { "enabled": true, "text": "……" }
}
```

- 改这个文件**不需要重启**：插件每一步组装提示词时都重新读盘。
- 想临时关掉某一段：把 `enabled` 改 `false`，或让模型调 `su_soul_toggle`。

### 4.2 基因组（`persona-memory/data/genome.json`）

```json
{
  "temperament": { "sensitivity": 1.0, "resilience": 1.0 },
  "molecules": [
    { "id": "dopamine", "name": "多巴胺", "baseline": 0.22, "half_life": 90 }
  ]
}
```

- `sensitivity`：事件来时飙多狠（0.5–2.0）
- `resilience`：好得多快（0.5–2.0）
- `molecules[].half_life`：半衰期（秒），越大越"记仇"
- ⚠️ **顺序必须稳定**：历史状态按位置存，顺序一变就对不上人。新分子只能追加在末尾。
- 改完需要重启 persona-memory。

### 4.3 路径与端口

插件端（`bootstrap.js` 的 `buildServices`）：

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `persona-memory` 端口 | `8772` | 改这里要同步改 `client.js` 的 `DEFAULT_BASE` |
| Mnemosyne 数据目录 | `<workspace>/_inspect/mnemosyne-data` | |
| Python 解释器 | `<workspace>/_inspect/mnemosyne/service/.venv/Scripts/python.exe` | |
| Milvus URI | 自动探测（WSL IP） | 环境变量 `MILVUS_URI` 优先 |
| 构建戳 | `EXPECTED_BUILD` | 改了 Python 代码就同步改它 + `api.py` 的 `SERVICE_BUILD`，下次 DSH 启动会自动重启服务 |

> **构建戳是干嘛的**：它让"服务该不该重启"变成一个可判断的问题——版本对不上就自己重启，不用人手动杀进程。

---

## 5. 首次启动与验证

### 5.1 启动

正常打开 DSH 即可。插件会在启动时：

1. 探测 persona-memory / mnemosyne 是否健康
2. 不健康就拉起（detached，脱离 DSH 进程树）
3. 比对构建戳，版本不对就重启服务

### 5.2 验证清单

```powershell
# 1. 服务活着
curl http://127.0.0.1:8772/health
# 期望：{"ok": true, "build": "...", "vector_up": true/false, ...}

# 2. 自检
curl http://127.0.0.1:8772/selfcheck
# 期望：空库时 ok=true，issues=[]

# 3. 人格三层（空库时是出厂值）
curl http://127.0.0.1:8772/persona
# 期望：base.warmth=0.85, shaped 各值等于 base, evidence_count=0

# 4. 缺口池空
curl http://127.0.0.1:8772/gaps
# 期望：{"open_count": 0, "count": 0}

# 5. 兴趣空
curl http://127.0.0.1:8772/interests
# 期望：{"count": 0, "confirmed_count": 0}

# 6. 看板
# 浏览器打开 http://127.0.0.1:8772/ui
```

### 5.3 "干净"的判定标准

部署后应当满足：

| 检查项 | 干净值 |
| --- | --- |
| `data/persona_state.json` | **不存在**，或存在但 `shaped.*.evidence_count == 0` |
| `data/gaps.json` | 不存在，或 `gaps: []` |
| `data/interests.json` | 不存在，或 `topics: []` |
| `data/presence.json` | 不存在（`has_history: false`） |
| `data/usage.json` | 不存在，或 `entries: {}` |
| `mnemosyne-data/data/memory.db` | 新建空库（`memories: 0`） |
| Milvus collection | 空（`rows: 0`） |

> 如果 `persona_state.json` 已存在且带了别人的数据，**删掉它**再启动——服务会用出厂 profile 重建。

---

## 6. 数据目录说明（重要）

| 文件 | 类型 | 能否随包分发 |
| --- | --- | --- |
| `dsh-suyi/persona.json` | 出厂设定（人格三段） | ✅ 可以 |
| `persona-memory/data/genome.json` | 出厂设定（分子/气质） | ✅ 可以 |
| `data/persona_state.json` | **用户数据**（人格状态） | ❌ 不可 |
| `data/gaps.json` | **用户数据**（执念池） | ❌ 不可 |
| `data/interests.json` | **用户数据**（兴趣画像） | ❌ 不可 |
| `data/presence.json` | **用户数据**（在场感） | ❌ 不可 |
| `data/usage.json` | **用户数据**（使用账本） | ❌ 不可 |
| `_inspect/mnemosyne-data/` | **用户数据**（记忆库） | ❌ 不可 |
| `dsh-suyi/data/*.log` | 运行日志 | ❌ 不可 |

**打包发布时**：只带源码 + `persona.json` + `genome.json`（可选示例），其余一律不带。

---

## 7. 常见问题

**Q：看板数据不动，是不是断联了？**
先确认看板打开的端口和服务端口一致（`http://127.0.0.1:8772/ui`）。历史上出现过同时跑多个实例、看板连到旧端口的情况。

**Q：`vector_up: false` 怎么办？**
1. 查 Milvus 是否在跑：`curl http://<milvus-host>:19530`（gRPC 端口会返回 404，属正常）
2. WSL 场景：`wsl -d Ubuntu-22.04 -- ps aux | grep milvus`
3. 确认 `MILVUS_URI` 指向正确地址
4. 重启 persona-memory

**Q：Milvus 集合行数归零了？**
WSL 重启可能丢索引。跑：
```powershell
cd persona-memory
python rebuild_index.py
```
数据在 SQLite（权威源），不会丢。

**Q：改了 Python 代码怎么让它生效？**
1. 改 `api.py` 的 `SERVICE_BUILD`
2. 改 `bootstrap.js` 的 `EXPECTED_BUILD` 为同一个值
3. 重启 DSH —— 构建戳对不上会自动重启服务

**Q：端口被占 / 起不来？**
```powershell
netstat -ano | findstr :8772
```
找到 PID 后结束它。若进程是管理员权限、普通终端杀不掉，用管理员 PowerShell 或重启电脑。

**Q：人格越来越"人机味"？**
这不是配置问题，是语料问题。可以：
- 在 `persona.json` 的 `personality` 里写"说话的样子"（少用表格、别用"首先/其次/总之"）
- 用 `su_evidence` 给"表达"这件事积累证据

---

## 8. 升级与备份

### 备份

要备份的只有三样：

1. `persona-memory/data/`（人格状态、执念、兴趣、在场感、账本）
2. `_inspect/mnemosyne-data/`（记忆库，SQLite）
3. Milvus 数据卷（可选——能从 SQLite 重建）

### 升级

1. 停 DSH（托盘退出，不是关窗口）
2. 备份上表三项
3. 替换源码
4. 起 DSH，看 `selfcheck` 是否 `ok: true`

### 回滚

源码目录保留旧版本即可；数据是向后兼容的（新字段读不到就退回默认值）。

---

## 9. 最小可用验证脚本

保存为 `verify-install.ps1`：

```powershell
$base = "http://127.0.0.1:8772"
$fail = 0

function Check($name, $ok, $detail) {
  if ($ok) { Write-Host "OK   $name" -ForegroundColor Green }
  else { Write-Host "FAIL $name — $detail" -ForegroundColor Red; $script:fail++ }
}

$h = Invoke-RestMethod "$base/health" -TimeoutSec 5
Check "service" $h.ok $h
Check "build stamp" ($h.build -ne "") $h.build

$sc = Invoke-RestMethod "$base/selfcheck" -TimeoutSec 5
Check "selfcheck" $sc.ok ($sc.issues -join '; ')

$p = Invoke-RestMethod "$base/persona" -TimeoutSec 5
Check "base anchored" ($p.base.warmth.value -eq 0.85) $p.base.warmth.value
$clean = ($p.shaped.PSObject.Properties | Where-Object { $_.Value.evidence_count -gt 0 }).Count -eq 0
Check "clean persona (no evidence)" $clean "has evidence"

$g = Invoke-RestMethod "$base/gaps" -TimeoutSec 5
Check "empty gaps" ($g.open_count -eq 0) $g.open_count

$i = Invoke-RestMethod "$base/interests" -TimeoutSec 5
Check "empty interests" ($i.count -eq 0) $i.count

Write-Host ""
if ($fail -eq 0) { Write-Host "ALL PASS — installation is clean." -ForegroundColor Green }
else { Write-Host "$fail check(s) failed." -ForegroundColor Red; exit 1 }
```

---

## 10. 一句话

> 装完它，你会得到一个**空的**、但已经会呼吸的系统：
> 人格是出厂值，记忆是空的，但它已经在等第一句话了。
