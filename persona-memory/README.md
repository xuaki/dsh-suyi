# persona-memory

**DSH 的「人格 + 记忆」融合层**：让记忆可以追溯、可以纠错，让人格稳定但不僵死。

> 起点是两个问题：把记忆放进「一直在训练的权重」会怎样？把人格「过拟合」会怎样？
> 上一轮读 `flybrain-connectome-llm` 得出的答案是：结构不是答案，量和质才是；
> 而一个不能说出「我为什么记得这件事」的记忆，不配做长期陪伴的底座。
> 这个项目就是那个结论的工程实现。

---

## 一、它由三块拼成

| 层 | 来源 | 承担什么 | 不承担什么 |
|---|---|---|---|
| **骨架** | [dsh-mnemosyne](https://github.com/DSH-Raising-Plan/dsh-mnemosyne)（REST 调用） | 观察/情节/候选/长期记忆、证据、状态机、SQLite 权威源 | 语义检索（它的 `vector_index` 空着） |
| **轴** | **milvus-lite**（嵌入式本地文件）或 **内存向量索引** | 向量召回，补上那个空位 | 不做权威源，丢了不致命 |
| **体温** | [dsh-plastic-memory](https://github.com/kalozamf233/dsh-plastic-memory)（MIT，逻辑移植） | 15 种分子、情绪构建、动态状态层 | **不承载人格**（原项目用学习率×3 做过拟合，这里否决了） |

人格在这里是**三层**，不是一张卡、也不是一组权重：

| 层 | 内容 | 更新节奏 | 硬约束 |
|---|---|---|---|
| **基础人格** | 性格倾向、表达风格、价值边界 | 基本不动 | 日常调用改不了，只有 `explicit_user_command=True` 才写 |
| **经验塑造** | 兴趣、熟悉度、习惯 | 缓慢 | ≥3 条独立证据才动手；单次位移 ≤0.05；向基础锚收缩 |
| **动态状态** | 情绪、注意力、疲劳 | 分钟到小时 | 分子指数衰减回基线，不写入长期 |

---

## 二、一次交互里发生了什么

```
你说一句话
   │
   ├─ ① 乱码闸门：U+FFFD / 只剩问号 → 直接拒绝（乱码进库就永远洗不掉）
   ├─ ② 情绪先于事实：识别情绪线索 → 释放分子 → 更新动态状态层
   ├─ ③ 准入：价值/可靠性/新颖度/持续性/明确反馈，任一达标才进认知链
   │        「嗯」「好」→ 拒；"记住…" → 直通
   ├─ ④ 写权威源：mnemosyne 存 Observation（可追溯）→ 候选记忆（不是 Active）
   ├─ ⑤ 写索引：Milvus 存向量（可失败，不阻塞）
   └─ ⑥ 返回：observation_id / memory_id / 是否准入 / 当时的情绪上下文

你问一句话
   │
   ├─ ① 若索引有脏数据，先 flush 一次（只在这里付代价）
   ├─ ② Milvus 语义召回 + mnemosyne 权威检索
   ├─ ③ 融合打分：语义 × 状态 × 来源 × 显著性 × 唤醒度
   ├─ ④ 过滤 INVALIDATED/REJECTED/DECAYED（纠错过的不得再被当正依据）
   ├─ ⑤ 近重复聚合（同一句话可能同时以 working/episodic 存在）
   └─ ⑥ 按 token 预算裁剪，返回带来源的片段 + 降级标记
```

---

## 三、快速开始

前置：本机已有
- `mnemosyne` 服务（本项目用 8765 跑通）
- `milvus-lite`（嵌入式，纯本地文件，不需要 Docker）

```powershell
# 1) mnemosyne（在它的 service 目录下）
$env:PYTHONPATH = "src"
python -m mnemosyne.main --host 127.0.0.1 --port 8765 --token dev-token --data-dir <数据目录>

# 2) 本服务
$env:PYTHONPATH = "persona-memory"
python -m persona_memory.main --port 8772
```

```powershell
# 说一句话
curl -X POST http://127.0.0.1:8772/observe -H "Content-Type: application/json" `
  -d '{"text":"我最近有点焦虑","source_kind":"user_statement","tier":"engage"}'

# 问一句话
curl -X POST http://127.0.0.1:8772/retrieve -H "Content-Type: application/json" `
  -d '{"query":"我为什么焦虑","limit":5}'

# 看人格三层
curl http://127.0.0.1:8772/persona
```

> ⚠️ **Windows 上别用 `Invoke-WebRequest` 直接发中文**：它默认按 UTF-16 发 body，
> 服务端会收到 `????`。用 `curl.exe`、Python `requests`，或显式把 body 编成 UTF-8 字节。
> 我踩过这个坑，所以服务端加了乱码闸门兜底。

---

## 四、接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 双服务健康 + 向量层状态 |
| POST | `/observe` | 记一条观察（含准入判定） |
| POST | `/retrieve` | 混合检索 + 近重复聚合 + 降级标记（默认**摘要模式**，`detail=true` 才给全文；`since_hours>0` 只回时间窗内） |
| POST | `/expand` | 链式召回：按 `memory_id` 展开原文与一跳邻居摘要（`depth` 1–2） |
| GET | `/proactive` | 主动提起：挑一件「此刻值得想起来」的旧事（带冷却期，别频繁调） |
| GET | `/persona` | 人格三层快照（含 15 分子浓度） |
| POST | `/persona/emotion` | 手动释放分子 |
| POST | `/feedback` | 用户纠正 → 失效 + 移出索引 |
| POST | `/promote` | 候选 → 长期（跨天 + engage/explicit 才允许） |
| GET | `/stats` | 汇总 |

---

## 五、几个关键决策（以及为什么）

1. **不训练记忆，也不训练人格。**
   `flybrain-connectome-llm` 的负结果：度保持打乱接线后质量差异 0.0016（只有 seed 散布的 0.04×）。
   结构本身没被证明有用，而训练式记忆不可审计、不可单条纠错。所以记忆用**证据链 + 索引**，
   人格用**慢变量 + 状态机**。

2. **Milvus 只是索引，不是权威源。**
   SQLite（mnemosyne）是唯一真相。Milvus 挂了，检索降级到 mnemosyne 仍可用；
   反过来不成立，所以写入顺序永远是「先权威，后索引」。

3. **不做"过拟合人格"。**
   过拟合会把事实一起焊死，而且不可更新。经验层用的是「高锚定、低更新速率」：
   证据不够不动、单步位移有上限、目标值先向基础锚收缩。

4. **情绪只调参，不翻盘。**
   分子浓度可以影响上下文预算、检索显著性、主动提起权重，
   但不能把未达门槛的记忆推进 ACTIVE——状态机说了算。

5. **乱码必须拒绝，不能"容错"。**
   一次 PowerShell 编码事故让 `????` 永久进了记忆库，提醒：写入闸门要比读取容错更严。

---

## 五点半、链式召回与 Inline & Spill

借鉴 [Super-agent 设计探索](https://www.bilibili.com/opus/1265361021143351318) 里的两个机制：

**Inline & Spill**：窗口里只放「摘要 + 引用 id」，全文留在服务端，需要时再拿。
实测同一条查询：摘要模式 54 字 / 全文模式 861 字——同一份 token 预算能装 16 倍。

**链式召回（Chain of Recall）**：每条记忆的 `metadata.links` 里存
`{rel, target_id, summary}`，构成一张「从现在指向过去」的引用图。

- **建图**：`observe` 写入时自动找邻居，建两类边：
  - `related_to`：内容像。先用向量层粗筛 30 条（召回要宽），
    再用 bigram Jaccard 重排、阈值 0.5（排序要准）。向量分只用来捞候选——
    hashing 向量对中文的**绝对分不可信**（无关句也能到 0.75），但**相对序**有用。
  - `temporal_neighbor`：时间上紧挨着（默认 5 分钟内），基本就是同一轮对话里
    你说一句、我回一句。内容可能一个字都不像，但它们属于同一条线（文章里的 Thread）。
- **召回**：`retrieve` 默认只带回 links 的 id + 一句摘要；
- **展开**：`expand(memory_id, depth=1..2)` 才把正文和邻居拉回来；
- **时间窗**：`since_hours>0` 只回这个窗口内的记忆。「我们昨天聊了什么」这类问题，
  靠时间过滤比靠语义猜准得多（插件侧的 `su_recall(recent_days=N)` 会换算成它）。
- **时间分层**：排序分带年龄衰减——近 7 天不衰减，7-30 天线性降到 0.85，
  再往后缓降但有下限 0.6。**记忆「记得住」和「先被想起」是两件事**：
  衰减只压后者，不会让老记忆沉底消失。
- **存量**：`python backfill_links.py --apply` 给历史记忆补图（可重复跑，不追加；
  语义边 + 时间邻居边，双向建），覆盖率能从 40% 提到 96%+。

> 顺带修了两个隐藏很久的 bug：
>
> 1. **Milvus 在 COSINE 下返回的是相似度、不是距离**。原代码写 `1.0 - score`，
>    等于「越像越扣分」——向量召回一直在反向加分，只是被 mnemosyne 的权威基线盖住了。
> 2. **mnemosyne 在 `capture_observation` 时就把 observation 自动转成 memory 了**
>    （status 直接 active，metadata 只有 `from_observation`），于是 `observe` 几乎总走
>    「复用已有」分支，那条自动记忆的 metadata 里 **tier / strength 从来没被写进去过**
>    （晋升因此只能按默认 attention 处理）。现在两个分支都会 merge 补写。

**主动提起（Proactive Recall）**：不问自答的那种。

`GET /proactive` 挑一件「此刻值得想起来」的旧事。规则和检索相反：

- 优先**用户说过的**（跳过我的长回复、跳过系统噪音、跳过 assistant 的回复）
- 必须在时间窗内：3 小时内 = 刚说过，30 天前 = 怕过时
- 提过的同一条冷却 24 小时——不重复，才不会变成唠叨

插件侧拿到的是一句摘要 + 「为什么现在想起」，注入系统提示词一次
（`suyi_recall`），提不提、怎么提由模型自己判断。

测试：`python test_links.py`（纯逻辑、不依赖任何服务）。

## 六、三臂对照实验结果

详见 [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)。核心数字（10 条旧记忆 + 灌入 10 条新记忆后回查）：

| arm | 抗遗忘 | 可溯源 | 延迟（中位） | 单条撤销 | 纠错有效 |
|---|---|---|---|---|---|
| A_trained（权重式，3 原型槽） | **0/5** | 0.0 | 0.25 ms | ❌ | ❌ |
| B_vector（纯向量） | 5/5 | 0.0 | 3.29 ms | ✅ | ✅ |
| C_fusion（本层） | 5/5 | **1.0** | 7.34 ms | ✅ | ✅ |

一句话：**权重式记忆的遗忘是容量与覆盖的必然；纯向量记得住但说不出「从哪来」；融合层两者都要，代价是 4 ms。**

## 七、测试

```powershell
python acceptance.py     # 一次跑完全部套件 + 数据卫生检查
```

单独跑：

```powershell
python test_smoke.py     # 40 项：分子 / 人格三层 / 准入与状态机（不需要服务）
python test_emotion.py   # 情绪误判修正：睡醒 / 失眠 / 否定式
python test_persist.py   # 人格持久化：重启、证据、分子、损坏文件回退
python test_e2e.py       # 31 项：mnemosyne + Milvus + 服务编排端到端
python test_api.py       # 12 项：REST 接口与中文往返
python test_garbled.py   # 乱码 / 短串 / token 闸门
python experiment.py     # 三臂对照实验，产出 experiment_result.json
```

运维脚本：

```powershell
python cleanup_junk.py           # 看哪些是测试污染（dry-run）
python cleanup_junk.py --apply   # 真删（只保留真实对话白名单）
python rebuild_index.py          # 以 mnemosyne 为准重建 Milvus 索引（直读 SQLite，绕过 API 的 50 条上限）
python watchdog.py               # 记忆体检：索引一致性 / 元数据 / 重复 / 卡住的候选 / 人格漂移 / 链接覆盖
python watchdog.py --fix         # 顺手修能安全修的（只补元数据，不删不合并）
python backfill_links.py --apply # 给存量记忆补引用图（链式召回）
python dedupe.py --apply         # 清理重复记忆（同内容多条；保留有 links/role 的那条）
python clean_noise.py --apply    # 清理系统噪音记忆（运行时上下文被误记进来的）
```

## 八、已知限制

- **默认向量器是 feature hashing**（`embedding.py`），零依赖、可离线，但语义能力有限。
  接口已抽象成 `Embedder`，接真句向量模型只需替换一个类。
- **mnemosyne 的 `/memories/search` 有硬上限 50 条**（`core/service.py` 里 `min(50, limit)`）。
  所以 `backfill_links.py` / `watchdog.py` 一次只能看到最近 50 条；
  要拿全量必须直读 SQLite（`rebuild_index.py` 已经这么做）。
  这一点真踩过：用 API 重建索引把 175 条记忆的向量缩成 50 行，靠 watchdog 才发现。
- **情绪识别是关键词匹配，不是理解。** 已修「睡醒了被读成要睡」「不开心被读成开心」，
  但换个说法（「一睁眼就中午」）它还是认不出来。这不是 bug 是能力边界。
- **mnemosyne 的 `idempotency_key` 在 hard delete 后仍会命中旧 id**，
  会导致「同一句话被删过就再也存不进去」。本项目已绕开：不依赖它的幂等键，
  改为先查精确匹配（只认活着的）再决定复用。
- **测试会写真实记忆库**。`test_e2e.py` / `test_api.py` 是往主库写的，
  跑完用 `cleanup_junk.py --apply` 清一次；正式使用应给测试独立 data-dir。
- **`/promote` 需要跨天数据**，单次会话里跑不出晋升——这是故意的。
- 项目当前与 mnemosyne 共用同一个 Python 环境（原型期）；
  正式部署应拆成各自 venv。
- 向量后端默认 `milvus-lite`（嵌入式）。**不再依赖 WSL / Docker**——Milvus 服务端连不上时会自动降级到内存索引。
