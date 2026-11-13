# -*- coding: utf-8 -*-
"""人格持久化：重启后人格不该归零。"""
from __future__ import annotations

import os
import sys
import tempfile

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


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="persona-persist-")
    path = os.path.join(tmp, "persona_state.json")

    # 1) 造一个有人格经历的状态
    svc = PersonaMemoryService(mnemosyne=None, vectors=None, persona_path=path)  # type: ignore[arg-type]
    svc.persona.release_emotion({"cortisol": 0.8, "oxytocin": 0.6})
    for i in range(3):
        svc.persona.observe_trait("warmth", delta=0.4, evidence_id=f"ev{i}")
    svc.persona.set_base("warmth", 0.9, explicit_user_command=True)
    svc.save_persona()
    check("state file written", os.path.exists(path), path)

    before = svc.persona.to_state_dict()
    cortisol_before = svc.persona.dynamic.endocrine.get("cortisol")
    warmth_before = svc.persona.shaped["warmth"].value
    ev_before = list(svc.persona.shaped["warmth"].evidence_ids)

    # 2) 模拟重启
    svc2 = PersonaMemoryService(mnemosyne=None, vectors=None, persona_path=path)  # type: ignore[arg-type]
    check("base survives restart", svc2.persona.base_value("warmth") == 0.9, str(svc2.persona.base))
    check(
        "shaped warmth survives",
        abs(svc2.persona.shaped["warmth"].value - warmth_before) < 1e-9,
        f"{svc2.persona.shaped['warmth'].value} vs {warmth_before}",
    )
    check("evidence ids survive", svc2.persona.shaped["warmth"].evidence_ids == ev_before)
    check(
        "molecules survive",
        abs(svc2.persona.dynamic.endocrine.get("cortisol") - cortisol_before) < 0.05,
        f"{svc2.persona.dynamic.endocrine.get('cortisol')} vs {cortisol_before}",
    )
    check("version survives", svc2.persona.shaped["warmth"].version >= 1)

    # 3) 损坏文件不该炸
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken json")
    svc3 = PersonaMemoryService(mnemosyne=None, vectors=None, persona_path=path)  # type: ignore[arg-type]
    check("corrupt state falls back gracefully", svc3.persona.base_value("warmth") == 0.85)

    print()
    if FAIL:
        print(f"RESULT: {FAIL} FAILED")
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
