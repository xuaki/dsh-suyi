# -*- coding: utf-8 -*-
"""两个「自动采集」入口的回归测试。

钉住 2026-11-14 评测里发现的两个空转机制：

  ① 兴趣画像：「我喜欢X」正则 481 条真实消息命中 0 条，su_interest_note 18 天 0 次；
  ② 失败账本：su_forget 18 天 0 次，/feedback 钩子从没被走过。

两个都是「建好了但没人喂」。现在 /observe 自动喂它们——
这个测试管的是「喂进来的东西别是垃圾」：宁漏不错。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persona_memory.correction_harvest import detect_correction  # noqa: E402
from persona_memory.interest_harvest import extract_implicit_topics  # noqa: E402

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

    print("[隐式兴趣：该抽出来的]")
    check("我最近又去滑雪了 → 滑雪", extract_implicit_topics("我最近又去滑雪了") == ["滑雪"])
    check("我还在玩吉他 → 吉他", extract_implicit_topics("我还在玩吉他") == ["吉他"])
    check(
        "我开始学摄影了，每天都要拍拍 → 摄影",
        extract_implicit_topics("我开始学摄影了，每天都要拍拍") == ["摄影"],
        str(extract_implicit_topics("我开始学摄影了，每天都要拍拍")),
    )
    check("我最近迷上了露营 → 露营", extract_implicit_topics("我最近迷上了露营") == ["露营"])
    check("我最近入坑了爬山 → 爬山", extract_implicit_topics("我最近入坑了爬山") == ["爬山"])
    check("一句里两个话题", extract_implicit_topics("我最近又去滑雪了，还在玩吉他") == ["滑雪", "吉他"])

    print("[隐式兴趣：必须抽不出的]")
    check("我最近有点累（无名词）", extract_implicit_topics("我最近有点累") == [])
    check("我今天在公司开会（无信号词）", extract_implicit_topics("我今天在公司开会") == [])
    check("我最近在想要不要辞职（纠结不是兴趣）", extract_implicit_topics("我最近在想要不要辞职") == [])
    check("我朋友最近又去滑雪了（不是他的事）", extract_implicit_topics("我朋友最近又去滑雪了") == [])
    check("他说他最近又去滑雪了（第三人称）", extract_implicit_topics("他说他最近又去滑雪了") == [])
    check("如果我最近开始学摄影（假设句）", extract_implicit_topics("如果我最近开始学摄影") == [])
    check("我最近在公司开会（介词短语）", extract_implicit_topics("我最近在公司开会") == [])
    check("我又忘了带钥匙（「又」+名词，但不是兴趣）", extract_implicit_topics("我又忘了带钥匙") == [])
    check("空串", extract_implicit_topics("") == [])

    print("[纠正识别：强模式]")
    r = detect_correction("你记错了，我没说过这个")
    check("你记错了，我没说过这个 → strong", bool(r) and r["strength"] == "strong", str(r))
    r = detect_correction("不是这样的，其实是周四")
    check("不是这样的，其实是周四 → strong", bool(r) and r["strength"] == "strong", str(r))
    r = detect_correction("你搞错了")
    check("你搞错了 → strong", bool(r) and r["strength"] == "strong", str(r))
    r = detect_correction("我没提过这件事")
    check("我没提过这件事 → strong", bool(r) and r["strength"] == "strong", str(r))

    print("[纠正识别：负模式 / 主语判别]")
    check("开玩笑的，你没记错 → 不触发", detect_correction("开玩笑的，你没记错") is None)
    check("你没记错，是我自己改了 → 不触发", detect_correction("你没记错，是我自己改了") is None)
    check("我是说，我没提过 → 不触发", detect_correction("我是说，我没提过这件事") is None)
    check("如果你记错了就改 → 不触发", detect_correction("如果你记错了就改") is None)
    check("你是不是这样的？→ 不触发（在问我）", detect_correction("你是不是这样的？") is None)
    # 「说错了」的主语是「我」= 他在改自己的口，不是纠正我
    check("不好意思我刚才说错了 → 不触发", detect_correction("不好意思我刚才说错了") is None)
    check("我记错了，其实是周四 → 不触发", detect_correction("我记错了，其实是周四") is None)
    check("我刚才看错了 → 不触发", detect_correction("我刚才看错了") is None)
    r = detect_correction("你刚才说错了")
    check("你刚才说错了 → strong", bool(r) and r["strength"] == "strong", str(r))
    r = detect_correction("说错了，是周四")
    check("说错了（无主语）→ strong", bool(r) and r["strength"] == "strong", str(r))
    r = detect_correction("宝宝你老是看错时间")
    check("你老是看错时间 → strong", bool(r) and r["strength"] == "strong", str(r))

    print("[纠正识别：弱模式]")
    r = detect_correction("不对，应该是周四")
    check("不对，应该是周四 → weak", bool(r) and r["strength"] == "weak", str(r))
    check("单说「不对」→ 不触发", detect_correction("不对") is None)

    print()
    if passed != total:
        print(f"RESULT: {total - passed} FAILED / {total}")
        return 1
    print(f"RESULT: ALL PASS ({passed}/{total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
