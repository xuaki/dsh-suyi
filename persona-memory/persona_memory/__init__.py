# -*- coding: utf-8 -*-
"""persona-memory：人格三层 + 证据记忆 的融合层。

上游项目都以「调用 / 移植」而非「整体改写」的方式使用：
- dsh-mnemosyne（AGPL-3.0-or-later）：记忆骨架，通过其 REST API 调用；
- dsh-plastic-memory（MIT）：分子/情绪调制，逻辑移植见 molecules.py。
"""

__version__ = "0.1.0"
