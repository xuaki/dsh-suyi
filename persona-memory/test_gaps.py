# -*- coding: utf-8 -*-
"""执念收割的回归测试。

钉住 2026-11-14 的那个真事故：
子智能体的报告通过 send_message 回到主会话，落成 role="user"，
整份报告被当成「他说的话」喂进收割 —— 报告里的列表标题
「四、不确定、我没动的（26 条，单列）」被收成执念，张力涨到 35，
连续两天每轮提示词都在注入它。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persona_memory.gaps import GapStore, _is_agent_relay, is_gap_worthy  # noqa: E402

passed = 0
total = 0


def check(name, cond, detail=""):
    global passed, total
    total += 1
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")


def main():
    global passed, total
    print("[agent relay detection]")
    check("Agent … sent a message 行", _is_agent_relay("Agent dc745f11-eb32 sent a message: 【报告】"))
    check("Background subagent 行", _is_agent_relay("Background subagent xxx finished and will do no further work"))
    check("正常对话不是 relay", not _is_agent_relay("我今天有点焦虑"))
    check("空行不是 relay", not _is_agent_relay("   "))

    print("[harvest: 报告整段不收割]")
    tmp = tempfile.mkdtemp(prefix="gaps_test_")
    path = os.path.join(tmp, "gaps.json")
    store = GapStore(path)

    report = """Agent dc745f11-eb32-4832-a9cd-d01c4ce34fae sent a message:

【记忆库清理完成｜软删 89 条】

四、不确定、我没动的（26 条，单列）
- 我不确定这条该不该删
- 我一直没弄明白为什么要保留它
"""
    n = store.harvest(report, role="user")
    check("报告里一行都不该被收割", len(n) == 0, f"实际收了 {len(n)} 条: {[g.text for g in n][:3]}")

    print("[harvest: 真人说的话照常收]")
    n2 = store.harvest("我一直没弄明白为什么 Milvus 老是挂，重启也没用", role="user")
    check("真人困惑能收（1 条）", len(n2) == 1, f"实际 {len(n2)} 条")
    if n2:
        check("收的是那句困惑", "Milvus 老是挂" in n2[0].text, n2[0].text)

    print("[harvest: assistant 不收割]")
    n3 = store.harvest("我一直没弄明白为什么这样", role="assistant")
    check("assistant 的话不收", len(n3) == 0)

    print("[is_gap_worthy 基本盘]")
    check("纯技术句不收", not is_gap_worthy("先把 rebuild_index 的路径改一下"))
    check("待办句不收", not is_gap_worthy("我接下来要做三件事"))
    check("带引号的不收", not is_gap_worthy("他说「我一直没弄明白」"))

    # 清理
    try:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    print(f"\nRESULT: {passed}/{total} PASS" if passed == total else f"\nRESULT: {passed}/{total} FAIL")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
