# -*- coding: utf-8 -*-
"""溯忆 · Watchdog：记忆健康的体检脚本。

对应 Super-agent 里的外部 Control Loop：主系统之外，定期看一眼
「有没有东西悄悄坏掉」。它不查端口存活（那是 bootstrap 的事），
而是查**语义层和一致性**：

  1. 索引一致性：Milvus 行数 vs 权威源活记忆数
  2. 缺失元数据：没有 tier 的记忆（历史遗留 → 晋升会走默认值）
  3. 重复记忆：内容几乎一样的多条（挤占召回预算，也让模型以为「说过很多次」）
  4. 卡住的候选：超过 N 天还停在 candidate
  5. 人格漂移：经验塑造层相对基础锚的偏移
  6. 链接覆盖：有多少记忆还没连进引用图（链式召回的有效率）

用法：
    python watchdog.py            # 只体检
    python watchdog.py --fix      # 顺手修能安全修的（只补元数据，不删不合并）
    python watchdog.py --json     # 机器可读输出
    python watchdog.py --stuck-days 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load(deps_root: Path | None = None):
    """延迟导入，保证 --help 在没有依赖时也能跑。"""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from persona_memory.mnemosyne_client import MnemosyneClient
    from persona_memory.milvus_store import MilvusVectorStore
    from persona_memory.persona import PersonaState
    from persona_memory.service import _similarity, _summarize

    return MnemosyneClient, MilvusVectorStore, PersonaState, _similarity, _summarize


DEAD = {"invalidated", "superseded", "rejected", "decayed", "archived"}
DUP_SIMILARITY = 0.9  # 内容相似度超过它 → 视为重复


def run(fix: bool = False, stuck_days: int = 7, as_json: bool = False, limit: int = 200) -> int:
    MnemosyneClient, MilvusVectorStore, PersonaState, _similarity, _summarize = _load()

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "findings": [], "fixes": []}
    mn = MnemosyneClient()

    def finding(level: str, code: str, message: str, data: dict | None = None) -> None:
        report["findings"].append({"level": level, "code": code, "message": message, "data": data or {}})

    # ---------------- 拉数据 ----------------
    try:
        rows = mn.search(query="", limit=limit)
    except Exception as e:  # noqa: BLE001
        report["error"] = f"mnemosyne 不可用: {e}"
        print(json.dumps(report, ensure_ascii=False, indent=1) if as_json else f"mnemosyne 不可用: {e}")
        return 2

    live = []
    for m in rows:
        if str(m.get("status", "")).lower() in DEAD:
            continue
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        live.append({"id": str(m.get("id")), "content": content, "status": str(m.get("status")),
                     "created_at": m.get("created_at")})
    report["scanned"] = len(live)

    # ---------------- 1. 索引一致性 ----------------
    # 注意：mnemosyne 的 search 有返回上限，`live` 只是本次扫描窗口内的样本；
    # 库内真实总数要看 health。拿窗口条数去比向量行数会得出假告警。
    total_memories = len(live)
    try:
        total_memories = int((mn.health() or {}).get("memories") or len(live))
    except Exception:  # noqa: BLE001
        pass
    try:
        v = MilvusVectorStore()
        v.connect()
        v.ensure_collection()
        vec_rows = int(v.stats().get("rows", 0))
        if vec_rows > total_memories * 1.2 + 10:
            finding("warn", "index_drift",
                    f"向量索引比库内记忆多 {vec_rows - total_memories} 行（可能有孤儿向量，跑 rebuild_index.py）",
                    {"vector_rows": vec_rows, "library_memories": total_memories})
        elif vec_rows < total_memories * 0.5 and total_memories > 20:
            finding("warn", "index_drift",
                    f"向量索引只有 {vec_rows} 行，库内记忆 {total_memories} 条（索引落后，跑 rebuild_index.py）",
                    {"vector_rows": vec_rows, "library_memories": total_memories})
        else:
            finding("ok", "index_drift",
                    f"索引覆盖正常（向量 {vec_rows} 行 / 库内记忆 {total_memories} 条；本次扫描窗口 {len(live)} 条）")
    except Exception as e:  # noqa: BLE001
        finding("warn", "vector_down", f"向量层不可用: {e}")

    # ---------------- 2. 缺失元数据 ----------------
    missing_meta = []
    for item in live:
        try:
            full = mn.get_memory(item["id"])
        except Exception:  # noqa: BLE001
            continue
        meta = full.get("metadata") or {}
        if "tier" not in meta:
            missing_meta.append(item["id"])
    if missing_meta:
        finding("warn", "missing_tier",
                f"{len(missing_meta)} 条记忆没有 tier（晋升会走默认 attention）",
                {"sample": missing_meta[:5]})
        if fix:
            fixed = 0
            for mid in missing_meta:
                try:
                    full = mn.get_memory(mid)
                    meta = dict(full.get("metadata") or {})
                    meta.setdefault("tier", "attention")
                    meta.setdefault("tier_backfilled_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
                    mn.update_memory(mid, metadata=meta)
                    fixed += 1
                except Exception:  # noqa: BLE001
                    continue
            report["fixes"].append(f"补齐 {fixed} 条记忆的 tier")
    else:
        finding("ok", "missing_tier", "所有记忆都有 tier")

    # ---------------- 3. 重复记忆 ----------------
    dups = []
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            if _similarity(live[i]["content"], live[j]["content"]) >= DUP_SIMILARITY:
                dups.append((live[i]["id"], live[j]["id"], live[i]["content"][:30]))
    if dups:
        finding("info", "duplicates",
                f"{len(dups)} 对内容高度重复的记忆（建议合并或失效其一）",
                {"sample": [{"keep_or_drop": a, "other": b, "preview": c} for a, b, c in dups[:5]],
                 "count": len(dups)})
    else:
        finding("ok", "duplicates", "没有内容高度重复的记忆")

    # ---------------- 4. 卡住的候选 ----------------
    now = time.time()
    stuck = []
    for item in live:
        if item["status"].lower() != "candidate":
            continue
        created = item.get("created_at") or ""
        try:
            from datetime import datetime

            ts = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
        except Exception:  # noqa: BLE001
            continue
        days = (now - ts) / 86400
        if days >= stuck_days:
            stuck.append({"id": item["id"], "days": round(days, 1), "preview": item["content"][:30]})
    if stuck:
        finding("info", "stuck_candidates",
                f"{len(stuck)} 条候选记忆超过 {stuck_days} 天没晋升（看准入证据够不够）",
                {"sample": stuck[:5]})
    else:
        finding("ok", "stuck_candidates", f"没有卡住超过 {stuck_days} 天的候选记忆")

    # ---------------- 5. 人格漂移 ----------------
    try:
        svc_data = (Path(__file__).resolve().parent / "data" / "persona_state.json")
        if svc_data.exists():
            with open(svc_data, encoding="utf-8") as f:
                state = json.load(f)
            shaped = state.get("shaped") or {}
            base = state.get("base") or {}
            drifts = []
            for k, sval in shaped.items():
                bval = (base.get(k) or {}).get("value") if isinstance(base.get(k), dict) else base.get(k)
                sval_v = sval.get("value") if isinstance(sval, dict) else sval
                if bval is None or sval_v is None:
                    continue
                drifts.append((k, round(float(sval_v) - float(bval), 4)))
            max_drift = max((abs(d) for _, d in drifts), default=0.0)
            if max_drift > 0.25:
                finding("warn", "persona_drift",
                        f"经验塑造层最大偏移 {max_drift}（超过 0.25 值得看一眼）",
                        {"drifts": drifts})
            else:
                finding("ok", "persona_drift", f"人格漂移正常（最大 {max_drift}）")
    except Exception as e:  # noqa: BLE001
        finding("info", "persona_drift", f"读取人格状态失败: {e}")

    # ---------------- 6. 疑似矛盾 ----------------
    # 规则法：同主题（相似度中等）+ 一边肯定一边否定 → 疑似矛盾。
    # 只报告、不自动处理：矛盾该改哪条要人来判断，记忆层不做语义裁决。
    PAIRS = (
        ("喜欢", "讨厌"),
        ("喜欢", "不喜欢"),
        ("想要", "不想要"),
        ("爱", "不爱"),
    )
    contradictions = []
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i]["content"], live[j]["content"]
            sim = _similarity(a, b)
            if not (0.2 <= sim < 0.85):
                continue
            hit = next(
                ((p, n) for p, n in PAIRS if (p in a and n in b) or (n in a and p in b)),
                None,
            )
            if hit:
                contradictions.append(
                    {
                        "a": live[i]["id"],
                        "b": live[j]["id"],
                        "pair": hit,
                        "a_preview": a[:24],
                        "b_preview": b[:24],
                    }
                )
    if contradictions:
        finding(
            "info",
            "contradictions",
            f"{len(contradictions)} 对疑似矛盾（同一主题、一正一反，需要人判断）",
            {"sample": contradictions[:5], "count": len(contradictions)},
        )
    else:
        finding("ok", "contradictions", "没有检测到疑似矛盾")

    # ---------------- 7. 链接覆盖 ----------------
    linked = 0
    no_links = 0
    for item in live:
        try:
            full = mn.get_memory(item["id"])
        except Exception:  # noqa: BLE001
            continue
        links = (full.get("metadata") or {}).get("links")
        if links:
            linked += 1
        else:
            no_links += 1
    coverage = (linked / len(live) * 100) if live else 0.0
    finding("info", "link_coverage",
            f"链式召回覆盖 {linked}/{len(live)} 条（{coverage:.0f}%），{no_links} 条还是孤岛",
            {"linked": linked, "orphan": no_links, "coverage_pct": round(coverage, 1)})

    # ---------------- 输出 ----------------
    report["summary"] = {
        "warn": sum(1 for f in report["findings"] if f["level"] == "warn"),
        "info": sum(1 for f in report["findings"] if f["level"] == "info"),
        "ok": sum(1 for f in report["findings"] if f["level"] == "ok"),
    }

    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        icons = {"ok": "✅", "info": "ℹ️ ", "warn": "⚠️ "}
        print(f"溯忆体检 · {report['generated_at']} · 扫描 {report.get('scanned', 0)} 条活记忆\n")
        for f in report["findings"]:
            print(f"{icons.get(f['level'], '·')} {f['message']}")
        if report["fixes"]:
            print("\n已修：")
            for x in report["fixes"]:
                print(f"  · {x}")
        print(f"\n结果：{report['summary']['warn']} 个警告 / {report['summary']['info']} 条信息")
    return 0 if report["summary"]["warn"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="溯忆 · 记忆健康体检")
    ap.add_argument("--fix", action="store_true", help="修能安全修的（只补元数据）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--stuck-days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    return run(fix=args.fix, stuck_days=args.stuck_days, as_json=args.json, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
