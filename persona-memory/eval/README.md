# eval —— 检索体检（retrieval health-check）

给「溯忆」装的一把固定尺子：回答「我改的机制，真的让回应变好了吗」。

## 文件

- `questions.json` —— **held-out 评估集**（与调参无关）。覆盖：明确事实 / 隐式偏好 / 纯噪声 / 英文词 / 否定·纠正过的。**不要用它调参**。
- `eval_retrieval.py` —— 跑检索、算分、出报告。只读调用 `POST /retrieve`，不写服务端、不重启服务。
- `report.md` —— 每次运行自动生成（覆盖写）。头部带配置快照。

## 怎么跑

```powershell
$py = "python"   # 换成你的 venv python
Set-Location <你的仓库>/persona-memory/eval
& $py eval_retrieval.py
```

可选参数：`--base-url`（默认 `http://127.0.0.1:8772`）、`--limit`（默认 5）、`--token-budget`（默认 10000）。

## 怎么做「前后对比」

1. 记下当前 `report.md` 头部的**配置快照**（vector_engine、阈值常量、构建戳）。
2. 改一个机制（阈值 / 向量引擎 / 证据分级……）。
3. 重跑 `eval_retrieval.py`，**不要动 questions.json**。
4. 对比两次的：总分、覆盖率、假阳性名单、漏检名单。

> 铁律：分数只对「同一份 questions.json + 同一检索预算 + 同一服务」有意义。
> 这是「同一把尺子下的前后对比」，不是绝对好坏；预算不对等、问题集漂移，分数不可比。
>
> 已知限制：记忆库是活的（持续写入、人格动态态在变、索引可用性在变），
> 两次运行之间分数可能有小幅抖动。对比时看「假阳性/漏检名单」的变化，比盯百分点更可靠。
