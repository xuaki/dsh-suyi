# -*- coding: utf-8 -*-
"""内分泌分子层（动态状态层）。

移植自 dsh-plastic-memory/lib/nn/endocrine.mjs（MIT License, 原作者保留）。
15 种分子浓度 0..1，随时间指数衰减回基线；事件触发释放。
effects() 把分子状态映射成对记忆与人格的**真实调制量**，而不是提示词标签。

为什么放在这里：mnemosyne 自己的设计文档把人格分三层，
「动态状态层」需要的就是这一个引擎——它不是人格，是体温。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class Molecule:
    id: str
    name: str
    baseline: float
    half_life: float  # 秒


# 内置默认：基因组读不到时的退路。
# ⚠️ 顺序必须和 data/genome.json 保持一致——历史内分泌状态是按位置存的，
# 顺序一变就对不上人了（新增分子只能追加在末尾）。
_DEFAULT_MOLECULES: List[Molecule] = [
    Molecule("dopamine", "多巴胺", 0.22, 90),
    Molecule("noradrenaline", "去甲肾上腺素", 0.18, 60),
    Molecule("serotonin", "血清素", 0.30, 300),
    Molecule("cortisol", "皮质醇", 0.15, 600),
    Molecule("adrenaline", "肾上腺素", 0.10, 45),
    Molecule("oxytocin", "催产素", 0.25, 120),
    Molecule("testosterone", "睾酮", 0.20, 240),
    Molecule("estrogen", "雌激素", 0.20, 600),
    Molecule("progesterone", "孕激素", 0.18, 600),
    Molecule("endorphin", "内啡肽", 0.20, 90),
    Molecule("acetylcholine", "乙酰胆碱", 0.22, 30),
    Molecule("gaba", "GABA", 0.25, 60),
    Molecule("glutamate", "谷氨酸", 0.25, 30),
    Molecule("melatonin", "褪黑素", 0.12, 120),
    Molecule("prolactin", "催乳素", 0.15, 300),
]

_DEFAULT_TEMPERAMENT: Dict[str, float] = {"sensitivity": 1.0, "resilience": 1.0}


def _load_genome():
    """从 data/genome.json 读分子与气质；读不到 / 读坏就用内置默认。

    心情引擎不该因为一个 JSON 写错就停摆——所以任何异常都退回默认值。
    """
    path = os.environ.get("PERSONA_GENOME") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "genome.json"
    )
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return list(_DEFAULT_MOLECULES), dict(_DEFAULT_TEMPERAMENT)

    mols: List[Molecule] = []
    for row in data.get("molecules") or []:
        try:
            mols.append(
                Molecule(
                    str(row["id"]),
                    str(row.get("name") or row["id"]),
                    float(row.get("baseline", 0.2)),
                    float(row.get("half_life", 60)),
                )
            )
        except Exception:  # noqa: BLE001 —— 跳过坏行，别让一行写错毁掉整个体质
            continue
    if not mols:
        mols = list(_DEFAULT_MOLECULES)

    temp = dict(_DEFAULT_TEMPERAMENT)
    raw_temp = data.get("temperament") or {}
    for k in temp:
        try:
            temp[k] = max(0.1, min(5.0, float(raw_temp.get(k, temp[k]))))
        except Exception:  # noqa: BLE001
            pass
    return mols, temp


MOLECULES, TEMPERAMENT = _load_genome()

MOLECULE_INDEX: Dict[str, int] = {m.id: i for i, m in enumerate(MOLECULES)}

# 情绪 12 维（与 plastic-memory 的 memorynet EMOTION_DIMS 同序）
EMOTION_DIMS = [
    "valence", "arousal", "joy", "sadness", "anger", "fear",
    "disgust", "surprise", "calm", "anxiety", "trust", "love",
]


def _raw_emotion(levels: Mapping[str, float]) -> Dict[str, float]:
    """情绪 10 维的原始合成（不含 valence / arousal）。

    关键：这些是**绝对值**，而不同情绪的基线差得很远——
    因为多巴胺基线只有 0.22，「悲伤 = 0.6×(1-多巴胺)+…」在完全平静时
    也会算出 0.53。直接展示绝对值，等于让人以为它一直很难过。

    所以看板上要的是「相对自己平时偏了多少」，基线由 BASELINE_EMOTION 给出。
    """
    g = lambda k: float(levels.get(k, 0.0))  # noqa: E731
    return {
        "joy": 0.6 * g("dopamine") + 0.4 * g("endorphin"),
        "sadness": 0.6 * (1 - g("dopamine")) + 0.4 * g("prolactin"),
        "anger": 0.5 * g("testosterone") + 0.3 * g("adrenaline") - 0.4 * g("serotonin"),
        "fear": 0.55 * g("cortisol") + 0.45 * g("adrenaline"),
        "disgust": 0.4 * (1 - g("serotonin")) + 0.2 * g("cortisol"),
        "surprise": 0.5 * g("adrenaline") + 0.5 * g("glutamate"),
        # 平静必须与压力分子负相关：皮质醇/肾上腺素高的时候，生理上不可能是
        # 真正的平静。原公式只有正向项，于是「焦虑 +100 / 恐惧 +99 / 平静 +49」
        # 会同时挂在看板上——人不可能既恐惧又平静。这是模型缺陷，不是数据。
        "calm": (
            0.5 * g("gaba")
            + 0.3 * g("serotonin")
            + 0.2 * g("melatonin")
            - 0.45 * g("cortisol")
            - 0.25 * g("adrenaline")
        ),
        "anxiety": 0.7 * g("cortisol") + 0.2 * g("noradrenaline"),
        # 信任同理：被压力顶满的时候，「信任」不该还在高位
        "trust": (
            0.6 * g("oxytocin")
            + 0.2 * g("estrogen")
            + 0.2 * g("serotonin")
            - 0.25 * g("cortisol")
        ),
        "love": 0.5 * g("oxytocin") + 0.3 * g("prolactin") + 0.2 * g("endorphin"),
    }


_BASELINE_LEVELS: Dict[str, float] = {m.id: m.baseline for m in MOLECULES}
# 完全平静时的情绪原始值。界面上 0.5 表示「和平时一样」。
BASELINE_EMOTION: Dict[str, float] = _raw_emotion(_BASELINE_LEVELS)
# 偏离放大倍数：原始差 0.25 → 界面差 0.25（1:1 映射，避免提前顶格后分不出轻重）
EMOTION_GAIN = 1.0


class EndocrineSystem:
    """15 分子状态机。纯 Python，无依赖，可直接跑在服务进程里。"""

    def __init__(self, seed: Optional[Iterable[float]] = None) -> None:
        if seed is None:
            self.levels: List[float] = [m.baseline for m in MOLECULES]
        else:
            seed_l = list(seed)
            self.levels = [
                _clamp(seed_l[i]) if i < len(seed_l) else MOLECULES[i].baseline
                for i in range(len(MOLECULES))
            ]

    # --- 基础读写 ---
    def get(self, molecule_id: str) -> float:
        i = MOLECULE_INDEX.get(molecule_id)
        return 0.0 if i is None else self.levels[i]

    def set(self, molecule_id: str, value: float) -> bool:
        i = MOLECULE_INDEX.get(molecule_id)
        if i is None:
            return False
        self.levels[i] = _clamp(value)
        return True

    def release(self, molecule_id: str, amount: float) -> bool:
        """按比例向 1 靠近：x <- x + (1-x)*amount。"""
        i = MOLECULE_INDEX.get(molecule_id)
        if i is None:
            return False
        # 敏感度：同一件事，玻璃心的人飙得更高
        amount = max(0.0, min(1.0, amount * TEMPERAMENT.get("sensitivity", 1.0)))
        v = self.levels[i]
        self.levels[i] = _clamp(v + (1.0 - v) * amount)
        return True

    def release_many(self, releases: Mapping[str, float]) -> None:
        for mid, amt in releases.items():
            self.release(mid, amt)

    def relax(self, molecule_id: str, amount: float) -> bool:
        """向基线靠拢：x <- x + (baseline - x) * amount。

        用于「解除」类事件——比如「睡醒了」应当把褪黑素拉回基线，
        而不是继续往 1 推。没有这个方向，系统只会在一条轴上越走越远。
        """
        i = MOLECULE_INDEX.get(molecule_id)
        if i is None:
            return False
        amount = max(0.0, min(1.0, amount))
        base = MOLECULES[i].baseline
        self.levels[i] = _clamp(self.levels[i] + (base - self.levels[i]) * amount)
        return True

    def relax_many(self, relaxes: Mapping[str, float]) -> None:
        for mid, amt in relaxes.items():
            self.relax(mid, amt)

    def tick(self, dt_seconds: float) -> None:
        """时间推进：指数衰减回基线。"""
        if dt_seconds <= 0:
            return
        # 恢复力：好得快的人，半衰期更短
        resilience = max(0.1, TEMPERAMENT.get("resilience", 1.0))
        for i, m in enumerate(MOLECULES):
            k = math.exp(-dt_seconds / (m.half_life / resilience))
            self.levels[i] = _clamp(m.baseline + (self.levels[i] - m.baseline) * k)

    # --- 调制效应 ---
    def effects(self) -> Dict[str, object]:
        g = self.get

        learning_rate_mod = (
            1.0
            + g("cortisol") * 0.5
            + g("noradrenaline") * 0.4
            + g("glutamate") * 0.25
            + g("acetylcholine") * 0.15
            - g("gaba") * 0.3
            - g("melatonin") * 0.35
        )

        exploration = _clamp(
            g("dopamine") * 0.8 + g("glutamate") * 0.35 - g("serotonin") * 0.45
        )  # 0..1

        arousal = _clamp(
            0.35 * g("noradrenaline")
            + 0.35 * g("adrenaline")
            + 0.2 * g("cortisol")
            + 0.15 * g("glutamate")
            - 0.3 * g("melatonin")
            - 0.2 * g("gaba")
        )

        valence = _clamp_signed(
            0.4 * g("dopamine")
            + 0.35 * g("endorphin")
            + 0.3 * g("oxytocin")
            + 0.2 * g("serotonin")
            - 0.45 * g("cortisol")
            - 0.3 * g("adrenaline")
        )

        social_axis = _clamp_signed(g("oxytocin") - g("testosterone"))

        # 情绪 10 维：以「自己的基线」为 0.5 中性，只在偏离时升降。
        # 这样「悲伤 0.5」= 平静，而不是「天生就 0.53」的假警报。
        levels = {m.id: self.levels[i] for i, m in enumerate(MOLECULES)}
        raw = _raw_emotion(levels)
        emotion: Dict[str, float] = {
            k: _clamp(0.5 + (v - BASELINE_EMOTION[k]) * EMOTION_GAIN) for k, v in raw.items()
        }
        emotion["valence"] = (valence + 1) / 2
        emotion["arousal"] = arousal

        return {
            "learning_rate_mod": learning_rate_mod,
            "exploration": exploration,
            "arousal": arousal,
            "valence": valence,
            "social_axis": social_axis,
            "memory_salience": 0.5 + 0.5 * arousal,
            "emotion": emotion,
        }

    # --- 序列化 ---
    def to_list(self) -> List[float]:
        return list(self.levels)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            m.id: {
                "name": m.name,
                "level": round(self.levels[i], 4),
                "baseline": m.baseline,
            }
            for i, m in enumerate(MOLECULES)
        }


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp_signed(x: float) -> float:
    return max(-1.0, min(1.0, x))
