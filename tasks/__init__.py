"""Task-family registry.  Four families, ordered here only by name."""

from __future__ import annotations

from typing import Dict, List

from .base import Defect, Instance, TaskFamily
from .cleansing import CleanseFamily
from .emails import EmailFamily
from .invoices import InvoiceFamily
from .reports import ReportFamily

FAMILIES: Dict[str, TaskFamily] = {
    f.name: f for f in (InvoiceFamily(), EmailFamily(), CleanseFamily(), ReportFamily())
}

#: Verifiability score from the pre-registered rubric, normalised to [0, 1].
#: Computed from the three sub-scores (each 0-3) assigned blind to results.
def verifiability(fam: TaskFamily) -> float:
    return round(sum(fam.vis.values()) / 9.0, 3)


ORDER: List[str] = sorted(FAMILIES)

__all__ = ["FAMILIES", "ORDER", "verifiability", "TaskFamily", "Instance", "Defect"]
