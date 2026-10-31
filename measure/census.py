"""Defect census - the independent measurement pass.

This module is deliberately isolated: it imports the task definition and nothing
from ``agents``.  It never sees a prompt, a reasoning trace, or a token count; it
sees only (raw instance, produced text) and grades the text against the ground
truth that was planted when the instance was generated.

Because grading is deterministic Python, the census is reproducible and cannot
drift with a judge model.  The price is that it can only grade what the product
schema makes explicit; free prose is graded through the family's own rule-based
checks (see ``tasks/reports.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tasks.base import Defect, Instance, TaskFamily


def census(fam: TaskFamily, inst: Instance, product_text: str) -> Dict[str, Any]:
    product = fam.parse_product(product_text or "")
    defects: List[Defect] = fam.census(inst, product)
    counts = {"critical": 0, "major": 0, "minor": 0}
    for d in defects:
        counts[d.severity] = counts.get(d.severity, 0) + 1
    return {
        "parse_ok": product is not None,
        "n_defects": len(defects),
        "counts": counts,
        "blocking": counts["critical"] + counts["major"],
        "defects": [d.as_dict() for d in defects][:40],
        "checks": inst.n_checks,
    }


def accepted(counts: Dict[str, int]) -> bool:
    """Acceptance standard used for the *census* side: no critical or major defect."""
    return counts.get("critical", 0) == 0 and counts.get("major", 0) == 0
