#!/usr/bin/env python3
"""Self-test: the census must pass the canonical (gold) deliverable with zero
defects, and must flag deliberately corrupted deliverables.

No model is involved.  If this fails, the ground truth and the grader disagree and
no downstream number means anything.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from tasks import FAMILIES, ORDER                     # noqa: E402
from measure.census import census                     # noqa: E402


def corrupt(fam_name: str, obj: dict) -> dict:
    """Make one plausible error, per family."""
    import copy
    o = copy.deepcopy(obj)
    if fam_name == "invoices":
        o["dispositions"][0]["payable"] = round(o["dispositions"][0]["payable"] + 100, 2)
    elif fam_name == "emails":
        o["drafts"][0]["actions"] = [{"type": "refund", "amount": 999.0}]
        o["drafts"][0]["reply"] = "We have issued a full refund. R1 R2 R3"
    elif fam_name == "cleansing":
        o["records"][0]["city"] = "Wrongville"
        o["records"][0]["source_ids"] = o["records"][0]["source_ids"][:1]
    else:
        o["findings"]["total_cases"] = int(o["findings"]["total_cases"] * 1.2)
        o["summary"] = o["summary"] + " A further 4712 cases were escalated."
    return o


def main() -> None:
    bad = 0
    for fam_name in ORDER:
        fam = FAMILIES[fam_name]
        insts = fam.generate(6, 13)
        clean = 0
        caught = 0
        for inst in insts:
            res = census(fam, inst, inst.canonical)
            if res["n_defects"] == 0:
                clean += 1
            else:
                print(f"  [{fam_name}] gold rejected on {inst.iid}: {res['defects'][:3]}")
            obj = fam.parse_product(inst.canonical)
            bad_obj = corrupt(fam_name, obj)
            res2 = census(fam, inst, json.dumps(bad_obj))
            if res2["counts"]["critical"] + res2["counts"]["major"] > 0:
                caught += 1
        print(f"{fam_name:10s} gold accepted {clean}/{len(insts)}   "
              f"corruption caught {caught}/{len(insts)}")
        bad += (len(insts) - clean) + (len(insts) - caught)
    print("SELFTEST", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
