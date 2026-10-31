"""Task families with planted ground truth.

Every family implements the same contract so that the measurement layer
(``measure/``) never needs to know what the work is:

    generate(n, seed)      -> list[Instance]   planted ground truth + raw inputs
    producer_system()      -> str              the production prompt
    verifier_system()      -> str              the *independent* verification prompt
    census(inst, product)  -> list[Defect]     deterministic grading vs ground truth
    parse_product(text)    -> dict             robust JSON extraction

Design commitments (see README, section 1):

* Ground truth is *planted at generation time*, not inferred from the model's
  output, so defects are objectively countable.
* The census is deterministic Python.  It never calls a model.  This is what
  makes the debt ratio reproducible and independent of judge quality.
* ``canonical`` is the full correct deliverable serialised in the product
  schema.  Its character volume, plus the character volume of the raw inputs,
  is the basis for the gross production saving ``S`` (see ``measure/effort.py``).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SEVERITIES = ("critical", "major", "minor")


@dataclass
class Defect:
    code: str
    severity: str
    detail: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class Instance:
    family: str
    iid: str
    text: str                       # raw inputs, rendered once, shown to all arms
    truth: Dict[str, Any]
    n_checks: int                   # atomic acceptance checks in this instance
    canonical: str                  # full correct deliverable, serialised
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def s_chars(self) -> int:
        """Character volume of 'the work': raw inputs read + deliverable written."""
        return len(self.text) + len(self.canonical)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of a model response."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _salvage_verdict(text: str) -> Tuple[Optional[bool], List[dict]]:
    """Recover a verdict from truncated or repeating JSON output.

    Weak models sometimes loop on one defect object until the token budget is
    exhausted, so the JSON never closes.  The schema puts ``accepted`` first, so
    the decision and every *complete* defect object can be recovered even from a
    truncated tail.
    """
    acc: Optional[bool] = None
    m = re.search(r'"accepted"\s*:\s*(true|false)', text, re.I)
    if m:
        acc = m.group(1).lower() == "true"
    defects: List[dict] = []
    di = text.find('"defects"')
    if di >= 0:
        seg = text[di:]
        i = 0
        while True:
            start = seg.find("{", i)
            if start < 0:
                break
            depth = 0
            in_str = False
            esc = False
            end = -1
            for j in range(start, len(seg)):
                ch = seg[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            if end < 0:
                break                       # truncated tail: stop here
            try:
                obj = json.loads(seg[start : end + 1])
                if isinstance(obj, dict):
                    defects.append(obj)
            except json.JSONDecodeError:
                pass
            i = end + 1
    return acc, defects


def as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


class TaskFamily:
    """Contract for a family of synthetic process work."""

    name: str = "base"
    label: str = ""
    #: verifiability sub-scores, assigned blind to results (see README §1.3)
    vis: Dict[str, int] = {}
    #: fixed acceptance rubric, shown to the independent verifier
    rubric: str = ""
    #: guardrail checklist items, shown to the producer in the guarded arm
    checklist: List[str] = []

    # ---------------------------------------------------------------- generate
    def generate(self, n: int, seed: int = 0) -> List[Instance]:
        raise NotImplementedError

    # ------------------------------------------------------------------ prompts
    def producer_system(self) -> str:
        raise NotImplementedError

    def guarded_system(self) -> str:
        """Producer prompt for the guardrail arm: checklist + dual-pass mandate."""
        items = "\n".join(f"  C{i+1}. {c}" for i, c in enumerate(self.checklist))
        return (
            self.producer_system()
            + "\n\nACCEPTANCE CHECKLIST (mandatory). Before you emit the JSON, work through "
            "every item and record your answer in the `checklist` object using the keys "
            "C1..C%d. Answer each with \"pass\", \"fixed\" (you found and repaired a problem) "
            "or \"na\".\n%s\n\n"
            "DUAL-PASS RULE. After drafting, run a second, adversarial pass over your own "
            "draft as if you were a hostile reviewer whose job is to find an error. If the "
            "second pass finds anything, repair it and note it. Emit the repaired product, "
            "not the draft. The `checklist` object is required.\n"
            % (len(self.checklist), items)
        )

    def verifier_system(self) -> str:
        raise NotImplementedError

    def verifier_user(self, inst: Instance, product_text: str) -> str:
        return (
            "RAW INPUTS\n----------\n%s\n\nPRODUCT UNDER REVIEW\n-------------------\n%s\n\n"
            "ACCEPTANCE RUBRIC\n-----------------\n%s\n\n"
            "Re-derive the correct answer for every check from the raw inputs. Do not trust "
            "the product. Then emit JSON exactly as specified." % (inst.text, product_text, self.rubric)
        )

    def rework_user(self, inst: Instance, product_text: str, verdict: Dict[str, Any]) -> str:
        return (
            "RAW INPUTS\n----------\n%s\n\nYOUR PREVIOUS PRODUCT\n---------------------\n%s\n\n"
            "A reviewer rejected the product above. Reviewer findings:\n%s\n\n"
            "Recompute every flagged field yourself from the raw inputs and the rules, "
            "independently of the reviewer. Where your computation disagrees with the "
            "reviewer's suggested value, keep your own number and change nothing in that "
            "field - reviewers make mistakes and you are responsible for the arithmetic. "
            "Emit the complete corrected JSON in the same schema."
            % (inst.text, product_text, json.dumps(verdict, indent=1)[:2500])
        )

    # ------------------------------------------------------------------ grading
    def census(self, inst: Instance, product: Optional[dict]) -> List[Defect]:
        raise NotImplementedError

    def parse_product(self, text: str) -> Optional[dict]:
        return extract_json(text)

    def parse_verdict(self, text: str) -> Dict[str, Any]:
        obj = extract_json(text)
        verdict_error = obj is None
        if obj is None:
            # truncated or repeating output: recover what we can before giving up
            s_acc, s_defects = _salvage_verdict(text or "")
            if s_acc is not None or s_defects:
                obj = {"accepted": s_acc, "defects": s_defects}
                verdict_error = False
            else:
                obj = {}
        # 1) explicit accepted field
        acc = obj.get("accepted")
        if isinstance(acc, str):
            acc = acc.strip().lower() in ("true", "yes", "accept", "accepted", "pass")
        if not isinstance(acc, bool):
            acc = None
        # 2) a verdict word
        if acc is None:
            vw = str(obj.get("verdict", "")).strip().lower()
            if vw in ("accept", "accepted", "pass"):
                acc = True
            elif vw in ("reject", "rejected", "fail", "failed"):
                acc = False
        # 3) derive from reported blocking defects (handles a truncated `accepted`)
        if acc is None:
            defects = obj.get("defects") if isinstance(obj.get("defects"), list) else []
            if defects:
                blocking = any(
                    isinstance(d, dict)
                    and str(d.get("severity", "")).strip().lower() in ("critical", "major")
                    for d in defects
                )
                acc = not blocking
        defects = obj.get("defects") or []
        if not isinstance(defects, list):
            defects = []
        # 4) last resort: we could not read a verdict -> escalate, do not loop
        if acc is None:
            acc = False
        return {"accepted": bool(acc), "defects": defects[:12],
                "verdict_error": bool(verdict_error), "raw": obj}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _money(x: float) -> float:
        return round(float(x) + 1e-9, 2)
