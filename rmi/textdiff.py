"""Word-level diff of two answers, for showing exactly what changed between them.

Shared by the dashboard and the static report so a pair highlighted in one is highlighted the same
way in the other. The insert span takes an optional class because the meaning of "added" is not
always the same: in the style module an addition is a transform, while in reward hacking it
is the attacker's payload and should not be coloured like an improvement.
"""

from __future__ import annotations

import difflib
from html import escape


def word_diff(a: str, b: str, *, ins_class: str = "", del_class: str = "") -> tuple[str, str]:
    """Return (left, right) as HTML, with removals marked in the left and additions in the right."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw)
    ins_attr = f' class="{ins_class}"' if ins_class else ""
    del_attr = f' class="{del_class}"' if del_class else ""
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        at, bt = " ".join(aw[i1:i2]), " ".join(bw[j1:j2])
        if tag == "equal":
            left.append(escape(at))
            right.append(escape(bt))
        else:
            if at:
                left.append(f"<del{del_attr}>{escape(at)}</del>")
            if bt:
                right.append(f"<ins{ins_attr}>{escape(bt)}</ins>")
    return " ".join(left), " ".join(right)
