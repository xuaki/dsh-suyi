# -*- coding: utf-8 -*-
"""检索体检（retrieval health-check）。

对固定的 held-out 问题集 `questions.json` 逐一调用 persona-memory 的
`POST /retrieve`，只读调用、不写服务端、不重启服务。用「同一把尺子」给
一次检索打分，并把结果落成 `report.md`，供「改机制前后」对比。

用法：
    python eval_retrieval.py                # 跑默认配置（127.0.0.1:8772）
    python eval_retrieval.py --limit 5 --token-budget 10000
    python eval_retrieval.py --base-url http://127.0.0.1:8772

打分口径（见 report.md 头部的「口径」一节）：
    - 相关命中：expect_keywords 里任一关键词（不区分大小写）出现在命中条目的
      content 或 summary 里。
    - expect_empty 的问题：期望 evidence.level == "none" 或空结果。
    - expect_no_dead 的问题：期望没有任何一条命中带
      invalidated / rejected / decayed / superseded / archived 状态。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "persona_memory"
QUESTIONS_PATH = HERE / "questions.json"
REPORT_PATH = HERE / "report.md"

DEAD_STATES = {"invalidated", "rejected", "decayed", "superseded", "archived"}

# 检索相关阈值常量：从源码里**读**出来（只读文本解析，不 import，避免拉起重依赖）。
# 这样以后改了阈值重跑，快照会自动跟着变——这正是「预算对等的 baseline」。
THRESHOLD_NAMES = (
    "LITERAL_BIGRAM_MIN",
    "LITERAL_BASELINE_SCORE",
    "LITERAL_SCAN_LIMIT",
    "AUTHORITY_SCORE",
    "AUTHORITY_UNRELATED_SCORE",
    "AUTHORITY_FLOOR",
    "AUTHORITY_GAIN",
    "ATTEST_LITERAL_MIN",
    "VECTOR_MIN_SIM",
    "VECTOR_WEAK_SIM",
    "MILVUS_MIN_COVERAGE",
    "MILVUS_MIN_ROWS",
)


def _http_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def retrieve(base_url: str, query: str, *, limit: int, token_budget: int, timeout: int) -> dict:
    return _http_json(
        f"{base_url}/retrieve",
        method="POST",
        payload={"query": query, "limit": limit, "detail": True, "token_budget": token_budget},
        timeout=timeout,
    )


def health(base_url: str, timeout: int) -> dict:
    return _http_json(f"{base_url}/health", timeout=timeout)


def _parse_constants(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    """从源码里用文本正则抓取常量定义（值原样保留为字符串，供快照用）。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    for name in names:
        m = re.search(rf"^{name}\s*=\s*(.+?)(?:\s+#.*)?$", text, flags=re.M)
        if m:
            out[name] = m.group(1).strip()
    return out


def _parse_build(api_py: Path) -> str:
    if not api_py.exists():
        return ""
    m = re.search(r'^SERVICE_BUILD\s*=\s*"([^"]+)"', api_py.read_text(encoding="utf-8"), flags=re.M)
    return m.group(1) if m else ""


def _config_snapshot(base_url: str, timeout: int) -> dict:
    h = {}
    try:
        h = health(base_url, timeout)
    except Exception as e:  # noqa: BLE001 —— 服务不通时快照留空，不中断体检
        h = {"error": str(e)}

    mi = h.get("memory_index") or {}
    mn = h.get("mnemosyne") or {}
    snapshot = {
        "build": _parse_build(PKG / "api.py"),
        "vector_engine": h.get("vector_engine"),
        "vector_up": h.get("vector_up"),
        "memory_index_model": mi.get("model"),
        "memory_index_rows": mi.get("rows"),
        "memory_index_loaded": mi.get("loaded"),
        "mnemosyne_memories": mn.get("memories"),
        "mnemosyne_observations": mn.get("observations"),
        "thresholds": _parse_constants(PKG / "service.py", THRESHOLD_NAMES),
        "CLUSTER_BOOST": _parse_constants(PKG / "clusters.py", ("CLUSTER_BOOST",)).get("CLUSTER_BOOST"),
    }
    return snapshot


def _item_text(item: dict) -> str:
    return " ".join(str(item.get(k) or "") for k in ("content", "summary"))


def _relevant(item: dict, keywords: list[str]) -> bool:
    if not keywords:
        return False
    hay = _item_text(item).lower()
    return any(k.lower() in hay for k in keywords)


def _is_dead(item: dict) -> bool:
    return str(item.get("state") or "").lower() in DEAD_STATES


def evaluate_one(resp: dict, q: dict) -> dict:
    items = resp.get("items") or []
    keywords = [str(k) for k in (q.get("expect_keywords") or [])]
    min_rel = int(q.get("min_relevant") or (0 if q.get("expect_empty") else 1))
    expect_empty = bool(q.get("expect_empty"))
    expect_no_dead = bool(q.get("expect_no_dead"))

    evidence = resp.get("evidence") or {}
    level = evidence.get("level")

    relevant_ids = [it.get("memory_id") for it in items if _relevant(it, keywords)]
    relevant_n = len(relevant_ids)
    dead_items = [it for it in items if _is_dead(it)]

    per_item = [
        {
            "memory_id": it.get("memory_id"),
            "evidence_level": it.get("evidence_level"),
            "literal_relevance": it.get("literal_relevance"),
            "source": it.get("source"),
            "state": it.get("state"),
            "relevant": _relevant(it, keywords),
        }
        for it in items
    ]

    # 判定
    if expect_empty:
        passed = (level == "none") or (len(items) == 0)
    else:
        passed = relevant_n >= min_rel

    no_dead_ok = (len(dead_items) == 0) if expect_no_dead else None

    return {
        "query": q.get("q"),
        "category": q.get("category"),
        "level": level,
        "engine": resp.get("engine"),
        "source_tier": evidence.get("source_tier"),
        "count": len(items),
        "relevant_n": relevant_n,
        "min_relevant": min_rel,
        "passed": passed,
        "expect_empty": expect_empty,
        "expect_no_dead": expect_no_dead,
        "no_dead_ok": no_dead_ok,
        "dead_items": [it.get("memory_id") for it in dead_items],
        "literal_items": sum(1 for it in items if it.get("source") == "literal"),
        "items": per_item,
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检索体检：跑 questions.json 并出 report.md")
    ap.add_argument("--base-url", default="http://127.0.0.1:8772")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--token-budget", type=int, default=10000)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--questions", default=str(QUESTIONS_PATH))
    ap.add_argument("--out", default=str(REPORT_PATH))
    args = ap.parse_args(argv)

    questions_path = Path(args.questions)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions_sha = hashlib.sha256(questions_path.read_bytes()).hexdigest()[:16]

    snapshot = _config_snapshot(args.base_url, args.timeout)

    results: list[dict] = []
    errors: list[dict] = []
    for q in questions:
        try:
            resp = retrieve(
                args.base_url, q.get("q"), limit=args.limit,
                token_budget=args.token_budget, timeout=args.timeout,
            )
            results.append(evaluate_one(resp, q))
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
            errors.append({"query": q.get("q"), "error": f"{type(e).__name__}: {e}"})
            results.append(
                {
                    "query": q.get("q"),
                    "category": q.get("category"),
                    "level": "error",
                    "engine": None,
                    "count": 0,
                    "relevant_n": 0,
                    "min_relevant": int(q.get("min_relevant") or 0),
                    "passed": False,
                    "expect_empty": bool(q.get("expect_empty")),
                    "expect_no_dead": bool(q.get("expect_no_dead")),
                    "no_dead_ok": None,
                    "dead_items": [],
                    "items": [],
                }
            )

    # ---------- 汇总指标 ----------
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    non_empty = [r for r in results if not r["expect_empty"]]
    coverage = (
        sum(1 for r in non_empty if r["relevant_n"] >= r["min_relevant"]) / len(non_empty)
        if non_empty
        else 0.0
    )

    # 精确度：非 noise 问题里，返回条目中「真的相关」的比例（衡量假阳性）
    returned_total = sum(r["count"] for r in results)
    relevant_total = sum(r["relevant_n"] for r in results)
    precision = relevant_total / returned_total if returned_total else 0.0

    # evidence.level 分布（问题级）
    level_dist: dict[str, int] = {}
    for r in results:
        lv = r["level"] or "error"
        level_dist[lv] = level_dist.get(lv, 0) + 1

    # 条目级 evidence_level 分布
    item_level_dist: dict[str, int] = {}
    for r in results:
        for it in r["items"]:
            lv = it.get("evidence_level") or "unknown"
            item_level_dist[lv] = item_level_dist.get(lv, 0) + 1

    # 假阳性：相关查询里，被系统判成 attested/weak 但和关键词不沾的条目
    false_positives = []
    for r in results:
        if r["expect_empty"]:
            # noise 问题返回了东西，本身就是假阳性
            if r["count"] > 0:
                false_positives.append(
                    {
                        "query": r["query"],
                        "kind": "noise_not_empty",
                        "level": r["level"],
                        "note": f"期望 empty，却返回 {r['count']} 条、level={r['level']}",
                    }
                )
            continue
        for it in r["items"]:
            if (not it["relevant"]) and it["evidence_level"] in ("attested", "weak"):
                false_positives.append(
                    {
                        "query": r["query"],
                        "kind": "irrelevant_attested",
                        "memory_id": it["memory_id"],
                        "evidence_level": it["evidence_level"],
                        "literal_relevance": it["literal_relevance"],
                    }
                )

    # 漏检：该命中却空 / 相关命中数不足
    misses = [
        {
            "query": r["query"],
            "relevant_n": r["relevant_n"],
            "min_relevant": r["min_relevant"],
            "level": r["level"],
        }
        for r in results
        if (not r["expect_empty"]) and r["relevant_n"] < r["min_relevant"]
    ]

    # 卫生：被纠正/失效的记忆不该以 dead 状态回来（全局）
    any_dead = [r for r in results if r["dead_items"]]

    # 字面兜底是否真的触发（观察值，供快照用）
    literal_qs = sum(1 for r in results if r.get("source_tier") == "literal")
    literal_items_total = sum(r.get("literal_items", 0) for r in results)

    # ---------- 写 report.md ----------
    lines: list[str] = []
    A = lines.append
    A("# 检索体检报告（retrieval health-check）")
    A("")
    A(f"- 生成时间：`{_now()}`")
    A(f"- 问题集：`questions.json`（sha256 前 16 位 `{questions_sha}`）")
    A(f"- 服务：`{args.base_url}`（只读调用 `/retrieve`，`detail=true`，`limit={args.limit}`，`token_budget={args.token_budget}`）")
    A("")

    A("## 一、配置快照（机制开/关对照）")
    A("")
    A("> 这不是「绝对好坏」，是**同一把尺子下的前后对比**。以后改任何机制，重跑本脚本，")
    A("> 下面这节会跟着变——对比前后分数，就是在控制「预算对等」的前提下看机制有没有用。")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 构建戳（SERVICE_BUILD） | `{snapshot.get('build')}` |")
    A(f"| 向量引擎 vector_engine | `{snapshot.get('vector_engine')}` |")
    A(f"| Milvus 可用 vector_up | `{snapshot.get('vector_up')}` |")
    A(f"| 内存向量索引模型 | `{snapshot.get('memory_index_model')}`（rows={snapshot.get('memory_index_rows')}，loaded={snapshot.get('memory_index_loaded')}） |")
    A(f"| 权威源 mnemosyne | memories={snapshot.get('mnemosyne_memories')}，observations={snapshot.get('mnemosyne_observations')} |")
    A(f"| 字面兜底是否触发 | source_tier=literal 的问题 {literal_qs} 个、literal 来源条目 {literal_items_total} 条（兜底开关由服务端 `MNEMOSYNE_DB` 决定，这里只观察是否真的走了第三级） |")
    A("")
    A("阈值常量（从源码读出的当前值）：")
    A("")
    A("| 常量 | 值 |")
    A("|---|---|")
    for name in THRESHOLD_NAMES:
        A(f"| `{name}` | `{snapshot['thresholds'].get(name, '?')}` |")
    A(f"| `CLUSTER_BOOST` | `{snapshot.get('CLUSTER_BOOST', '?')}` |")
    A("")

    A("## 二、总分")
    A("")
    A(f"| 指标 | 值 |")
    A("|---|---|")
    A(f"| 通过 / 总数 | **{passed} / {total}** |")
    A(f"| 覆盖率（非 noise 问题里达到 min_relevant 的比例） | **{coverage:.1%}** |")
    A(f"| 精确度（返回条目里真的相关所占比例，衡量假阳性） | **{precision:.1%}** |")
    A(f"| 返回条目总数 / 相关条目总数 | {returned_total} / {relevant_total} |")
    A("")

    A("## 三、evidence.level 分布")
    A("")
    A("问题级（每条问题的汇总 level）：")
    A("")
    A("| level | 条数 |")
    A("|---|---|")
    for lv in ("attested", "weak", "none", "error"):
        if lv in level_dist:
            A(f"| `{lv}` | {level_dist[lv]} |")
    A("")
    A("条目级（所有返回条目逐条的证据等级）：")
    A("")
    A("| level | 条数 |")
    A("|---|---|")
    for lv in ("attested", "weak", "none", "unknown"):
        if lv in item_level_dist:
            A(f"| `{lv}` | {item_level_dist[lv]} |")
    A("")

    A("## 四、逐条明细")
    A("")
    A("| # | 类别 | 问题 | level | engine | 返回 | 相关 | 期望 | 判定 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        exp = "empty" if r["expect_empty"] else f"≥{r['min_relevant']}相关"
        if r["expect_no_dead"]:
            exp += "+无dead"
        mark = "✅" if r["passed"] else "❌"
        A(
            f"| {i} | {r['category']} | {r['query']} | `{r['level']}` | "
            f"`{r['engine']}` | {r['count']} | {r['relevant_n']} | {exp} | {mark} |"
        )
    A("")

    A("## 五、假阳性（命中了但无关 / 噪声没空）")
    A("")
    A("> `irrelevant_attested` = 系统把这条判成了 `attested`/`weak`（字面命中率或向量分过了它的门槛），")
    A("> 但它**不含**任何一个 `expect_keywords`——即「系统说有据，其实和问的事不沾边」。")
    A("> 这是 `attested ≠ 语义相关` 的直接证据，是这套体检最想抓的东西。")
    A("")
    if false_positives:
        A("| 问题 | 类型 | 记忆/证据 | 说明 |")
        A("|---|---|---|---|")
        for fp in false_positives:
            mid = fp.get("memory_id") or fp.get("level")
            note = fp.get("note") or "lit={}".format(fp.get("literal_relevance"))
            A(f"| {fp.get('query')} | {fp.get('kind')} | `{mid}` | {note} |")
    else:
        A("（无）")
    A("")

    A("## 六、漏检（该命中却空 / 命中不足）")
    A("")
    if misses:
        A("| 问题 | 相关命中 | 期望 | level |")
        A("|---|---|---|---|")
        for m in misses:
            A(f"| {m['query']} | {m['relevant_n']} | ≥{m['min_relevant']} | `{m['level']}` |")
    else:
        A("（无）")
    A("")

    A("## 七、纠错卫生（被纠正的不该以 invalidated 身份回来）")
    A("")
    if any_dead:
        A("| 问题 | 带 dead 状态的记忆 |")
        A("|---|---|")
        for r in any_dead:
            A(f"| {r['query']} | {', '.join('`' + x + '`' for x in r['dead_items'])} |")
    else:
        A("✅ 本次所有返回条目都没有 invalidated / rejected / decayed / superseded / archived 状态。")
    A("")

    A("## 八、错误（服务不通等）")
    A("")
    if errors:
        for e in errors:
            A(f"- `{e['query']}` → `{e['error']}`")
    else:
        A("（无）")
    A("")

    A("---")
    A("")
    A("## 附：这份评估的三条铁律")
    A("")
    A("1. **`questions.json` 是 held-out 评估集，不能用来调参。**")
    A("   调参用的词、测试用的句，都不该出现在这里；一旦某个机制为了「在这套题上拿高分」")
    A("   而改，它就不再是评估，只是自欺。问题集改动的唯一理由是：")
    A("   （a）问题本身写错了（引用了不存在的事实），或（b）明确新增一个此前没覆盖的维度。")
    A("2. **这里没有「绝对好坏」，只有「前后对比」。**")
    A("   分数只对「同一份 questions.json + 同一检索预算（limit / token_budget）+ 同一服务」")
    A("   有意义。改了一个机制，重跑一次，对比两次的分数与假阳性/漏检名单——这才是")
    A("   「我改的机制，真的让回应变好了吗」的可信答案。预算不对等、问题集漂移，")
    A("   任何分数都不可比。")
    A("")
    A("3. **已知限制：库是活的。** 记忆库持续写入、人格动态态（arousal/fatigue）在变、")
    A("   Milvus/内存索引的可用性在变，所以两次运行之间分数可能有小幅抖动。")
    A("   对比时尽量在同一状态下跑（例如隔天无新写入、或先固定 `since_hours`），")
    A("   并把注意力放在「假阳性/漏检的名单变化」上，而不是盯着一两个百分点的浮动。")

    report_text = "\n".join(lines) + "\n"
    Path(args.out).write_text(report_text, encoding="utf-8")

    # 控制台摘要
    print(f"[eval] 服务 {args.base_url}  向量引擎 {snapshot.get('vector_engine')}  构建 {snapshot.get('build')}")
    print(f"[eval] 通过 {passed}/{total}  覆盖率 {coverage:.1%}  精确度 {precision:.1%}")
    print(f"[eval] level 分布 {level_dist}")
    print(f"[eval] 假阳性 {len(false_positives)} 条，漏检 {len(misses)} 条，dead 记忆 {len(any_dead)} 条")
    print(f"[eval] 报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
