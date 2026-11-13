# 溯忆 · Suyi

> 给 AI 装上「人格」和「记忆」的那一层。

**溯忆**是一个 DSH（DeepSeek Harness）插件 + 独立 Python 服务，让人格与记忆成为可追溯、可审计、可演化的系统，而不是一段越写越长的提示词。

```
┌─────────────────────────────────────────────┐
│  DSH 插件（dsh-suyi）                        │
│   · 组装提示词：人格 / 状态 / 经验 / 执念      │
│   · 工具：su_recall · su_remember · su_evidence│
│           su_gap · su_interest · su_forget    │
└──────────────────┬──────────────────────────┘
                   │ HTTP（127.0.0.1:8772）
┌──────────────────▼──────────────────────────┐
│  persona-memory（Python 服务）               │
│   · 人格三层：基础 / 经验塑造 / 动态状态       │
│   · 记忆编排：语义 + 权威 + 字面 + 证据分级    │
│   · 兴趣 / 执念 / 失败账本 / 常驻画像          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  dsh-mnemosyne（外部依赖，需自己安装）        │
│   · SQLite 权威源 + 证据链                    │
└─────────────────────────────────────────────┘
```

## 它解决什么问题

大多数「AI 记忆」是把历史对话塞进上下文。那没法回答三个问题：

1. **它凭什么这么说？** —— 溯忆每条召回都带 `evidence.level`（`attested` / `weak` / `none`），能说「我没依据」。
2. **我纠正它，它真的改了吗？** —— 纠正会沿着证据链扣回人格变化量，而不是只改一条记忆。
3. **它是怎么变成现在这样的？** —— 人格三层：基础（只读锚点）/ 经验塑造（≥3 条独立证据才动，每次 ≤0.05）/ 动态状态（激素、情绪、疲劳，会衰减）。

## 核心特性

| 特性 | 说明 |
| --- | --- |
| **人格三层** | 基础人格只读；经验塑造要有 ≥3 条独立证据才动，单次最多挪 0.05，且被基础人格往回拉；动态状态用 15 种分子模拟情绪起伏 |
| **证据链记忆** | 记忆不是文本块，是带来源、置信、时序和图邻居的证据节点 |
| **诚实检索** | 返回 `evidence.level`，外加逐条 `literal_relevance`；不会假装「我记住了」 |
| **兴趣自动采集** | 从「我最近又去滑雪了」这种随口一提里嗅线索，跨 3 天重复才确认 |
| **失败账本** | 自动识别「你记错了」，记下翻过的车，下一轮防着 |
| **执念池** | 「我还没弄明白的事」会绷着，越久张力越高，到阈值自己浮进提示词 |
| **常驻画像** | 一块 ≤400 字的人类可读底稿，省掉大量检索 |
| **检索体检** | `eval/` 里有 held-out 问题集，改任何机制都能用同一把尺子前后对比 |

## 安装

### 前置

- **DSH**（DeepSeek Harness）—— 插件宿主
- **[dsh-mnemosyne](https://github.com/DSH-Raising-Plan/dsh-mnemosyne)** —— 记忆层后端，提供 SQLite 权威源与 REST API（端口 8765）
- **Python 3.12+** —— 跑 persona-memory
- **Node.js 20+** —— 跑插件

### 目录布局

溯忆按下面的布局找东西（也可以用环境变量改，见下）：

```
<你的工作区>/
├── dsh-suyi/                 ← 本仓库的 dsh-suyi
├── persona-memory/           ← 本仓库的 persona-memory
└── _inspect/
    ├── mnemosyne/service/    ← dsh-mnemosyne 的 Python 包（你自己装）
    └── mnemosyne-data/       ← mnemosyne 的数据目录（自动生成）
```

### 步骤

```powershell
# 1. 拿到本仓库，放进你的 DSH 工作区
# 2. 装 mnemosyne（按它自己的 README），确保能跑：
#    python -m mnemosyne.main --host 127.0.0.1 --port 8765 --token dev-token

# 3. 给 persona-memory 建 venv 并装依赖
cd persona-memory
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 4. 把插件装进 DSH
#    按 DSH 的插件安装方式注册 dsh-suyi 目录

# 5. 重启 DSH —— 插件会自己拉起 persona-memory
```

### 环境变量（可选）

路径不符合上面的布局时，用这些覆盖：

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `SUYI_MNEMOSYNE_ROOT` | mnemosyne 服务目录 | `<工作区>/_inspect/mnemosyne/service` |
| `SUYI_MNEMOSYNE_DATA` | mnemosyne 数据目录 | `<工作区>/_inspect/mnemosyne-data` |
| `SUYI_PERSONA_ROOT` | persona-memory 目录 | `<工作区>/persona-memory` |
| `SUYI_PERSONA_DATA` | persona-memory 数据目录 | `<SUYI_PERSONA_ROOT>/data` |
| `SUYI_PYTHON` | 跑服务的解释器 | `<SUYI_MNEMOSYNE_ROOT>/.venv/Scripts/python.exe` |
| `MILVUS_URI` | 向量后端（见下） | `data/milvus_lite.db` |

### 向量后端

**默认用 milvus-lite**（嵌入式，纯本地文件），不需要 Docker、不需要 Milvus 服务。

如果你有独立的 Milvus 服务，设：

```
MILVUS_URI=http://127.0.0.1:19530
```

Milvus 连不上也不会挂——会自动降级到**内存向量索引**（`data/vector_index.npz`，几百到几万条够用）。

## 数据都在本地

溯忆**不上传任何东西**。所有数据都在你机器上：

| 路径 | 内容 |
| --- | --- |
| `persona-memory/data/persona_state.json` | 人格三层状态与证据 |
| `persona-memory/data/{gaps,interests,failures,usage,presence}.json` | 执念 / 兴趣 / 失败账本 / 用量 / 在场感 |
| `persona-memory/data/vector_index.npz` | 内存向量索引（可删，会重建） |
| `persona-memory/data/milvus_lite.db` | milvus-lite 数据（可删，会重建） |
| `<工作区>/_inspect/mnemosyne-data/data/memory.db` | **记忆权威源**（唯一真相） |

> 本仓库**不包含任何用户数据**。`persona.json` 里是示例角色卡，可以随便改。

## 目录结构

```
dsh-suyi/                  DSH 插件（JS）
├── src/
│   ├── index.js           插件主体：注入段 + 工具 + 子会话分级
│   ├── bootstrap.js       拉起/守护 Python 服务
│   ├── client.js          persona-memory 的 HTTP 客户端
│   └── tool-spec.js       工具定义
├── persona.json           示例人格（三段：身份 / 性格 / 沉浸思考）
└── test/                  插件单测

persona-memory/            Python 服务
├── persona_memory/
│   ├── persona.py         人格三层与证据
│   ├── service.py         记忆编排（检索 / 分簇 / 链接 / 证据分级）
│   ├── api.py             FastAPI 路由
│   ├── gaps.py            执念池
│   ├── interests.py       兴趣画像 + 状态机
│   ├── failures.py        失败账本
│   ├── profile.py         常驻画像
│   ├── interest_harvest.py     隐式兴趣采集
│   ├── correction_harvest.py   纠正识别
│   ├── vector_memory.py   内存向量索引
│   ├── content_guard.py   写入安检（凭据 / 注入 / 外泄 / 不可见字符）
│   └── ...
├── eval/                  检索体检（held-out 问题集）
├── docs/                  部署 / 评测文档
└── test_*.py              回归测试
```

## 测试

```powershell
cd persona-memory
.\run_tests.ps1          # 全量
.\run_tests.ps1 -Fast    # 只跑纯逻辑（不联网）
```

## 评测

```powershell
cd persona-memory
python eval/eval_retrieval.py
```

> 需要 persona-memory 已经在跑（127.0.0.1:8772）—— 体检是真的去打 /retrieve。

会输出通过率 / 覆盖率 / 精确度 / 假阳性，并写一份 `eval/report.md`。
`eval/questions.json` 是 **held-out 集，不要拿来调参** —— 调参用的样本和它必须是两批。

## 设计取舍

有几个地方是**故意**这么做的：

- **Milvus 可选，内存索引兜底**：我们的量级用不上独立向量库，多一个组件就多一个会坏的地方。
- **中文用 jieba 整词，不用 2-gram**：2-gram 会把「今天天气」切成「天天」这种跨词片段，造成假命中。
- **兴趣单条永远不确认**：要跨 3 天重复，或明确句式。误记归属比漏记更糟。
- **纠正「宁漏不错」**：「不对，应该是…」这种弱信号只记录不落账；「开玩笑的」优先压制。
- **静默吞异常是故意避免的**：能降级就降级，但一定留日志——否则坏了只知道「数字是 0」，不知道「为什么是 0」。

## License

MIT
