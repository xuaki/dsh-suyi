# -*- coding: utf-8 -*-
"""情绪识别修正验证：「睡醒了」不该被读成「要睡了」。"""
from __future__ import annotations

import sys

from persona_memory.molecules import EndocrineSystem
from persona_memory.persona import PersonaState
from persona_memory.service import PersonaMemoryService

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    if cond:
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def fresh_service():
    class _NoMn:
        pass

    # 只测情绪识别，不碰网络：用一个空壳 service 调用 _detect_emotions
    svc = PersonaMemoryService.__new__(PersonaMemoryService)
    svc.persona = PersonaState.from_base_profile({"warmth": {"value": 0.85}})
    return svc


def main() -> int:
    svc = fresh_service()

    # 0) 平静基线：情绪维度应该都落在 0.5 附近，不该天生「悲伤 0.53」
    calm = EndocrineSystem().effects()["emotion"]
    print("平静基线 ->", {k: round(v, 3) for k, v in calm.items()})
    check("calm sadness is neutral", abs(calm["sadness"] - 0.5) < 0.02, str(calm["sadness"]))
    check("calm anxiety is neutral", abs(calm["anxiety"] - 0.5) < 0.02, str(calm["anxiety"]))
    check("calm joy is neutral", abs(calm["joy"] - 0.5) < 0.02, str(calm["joy"]))

    # 1) 睡醒了：应该解除褪黑素，而不是释放
    rel, rlx = svc._detect_emotions("老公中午睡醒了，声音有点哑")
    print("睡醒了 ->", rel, rlx)
    check("睡醒 releases nothing sleepy", "melatonin" not in rel, str(rel))
    check("睡醒 relaxes melatonin", rlx.get("melatonin", 0) > 0.5, str(rlx))
    check("睡醒 relaxes gaba", rlx.get("gaba", 0) > 0.5, str(rlx))

    # 2) 真困：要睡了 → 释放褪黑素
    rel2, rlx2 = svc._detect_emotions("我好困，想睡了")
    print("想睡了 ->", rel2, rlx2)
    check("sleepy releases melatonin", rel2.get("melatonin", 0) > 0.4, str(rel2))

    # 3) 睡不着：负面，不该释放褪黑素，该升皮质醇
    rel3, rlx3 = svc._detect_emotions("我最近失眠，睡不着")
    print("失眠 ->", rel3, rlx3)
    check("insomnia releases cortisol", rel3.get("cortisol", 0) > 0.4, str(rel3))
    check("insomnia does not boost melatonin as sleepy", rel3.get("melatonin", 0) < 0.35, str(rel3))

    # 4) 否定：「不开心」「不累」不该触发正向/负面线索
    rel4, _ = svc._detect_emotions("我不开心")
    print("不开心 ->", rel4)
    check("negated joy not released", "dopamine" not in rel4, str(rel4))

    rel5, _ = svc._detect_emotions("我不累")
    print("我不累 ->", rel5)
    check("negated tired not released", "cortisol" not in rel5, str(rel5))

    # 5) 正常正向仍然生效
    rel6, _ = svc._detect_emotions("太好了，我太开心了")
    print("开心 ->", rel6)
    check("positive still works", rel6.get("dopamine", 0) > 0.4, str(rel6))

    # 5.5) 引用 / 提问 / 第三人称，不该触发自己的情绪
    rel7, _ = svc._detect_emotions("现在你为什么 悲伤焦虑呢")
    print("询问（引用） ->", rel7)
    check("question does not trigger", not rel7, str(rel7))

    rel8, _ = svc._detect_emotions("他最近很焦虑")
    print("第三人称 ->", rel8)
    check("third person does not trigger", not rel8, str(rel8))

    rel9, _ = svc._detect_emotions("我好焦虑")
    print("自述焦虑 ->", rel9)
    check("self anxiety triggers", rel9.get("cortisol", 0) > 0.4, str(rel9))

    # 6) 端到端：走一次真实的体温更新
    svc.persona.release_emotion({}, relaxes={"melatonin": 0.9, "gaba": 0.9})
    eff = svc.persona.effects()
    print("解除后 arousal:", eff["arousal"])
    check("wake restores arousal", eff["arousal"] > 0.05, str(eff["arousal"]))

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
