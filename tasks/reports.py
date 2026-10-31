"""Family 4 - operational report summaries.

Two ten-week tables - cases resolved and average handling time - with planted
structure in both: a sustained volume decline in one region, a single-week volume
anomaly elsewhere, an aggregate trend, and a sustained handling-time increase in a
third region.  The product is a structured findings block plus a written summary
whose every quantitative claim is checkable against the tables.

This is the low-verifiability family.  Nothing about the deliverable is
under-determined *in principle*, but three things make checking expensive: the
claim set is chosen by the producer, so the reviewer cannot enumerate it in
advance; each claim has to be re-derived from a hundred and twenty cells; and a summary can be
numerically wrong in a way that reads perfectly well.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

from .base import Defect, Instance, TaskFamily, as_float

REGIONS = ["North", "South", "East", "West", "Central", "Coastal"]
FIELDS = ("total_cases", "worst_region", "worst_region_pct", "anomaly_week",
          "anomaly_drop", "trend", "aht_region", "aht_pct")


class ReportFamily(TaskFamily):
    name = "reports"
    label = "Operational report summaries"
    vis = {"ground_truth_checkability": 1, "rule_determinism": 1, "judgement": 0}
    rubric = (
        "Check the findings block and the written summary against the two tables.\n"
        "  (1) total_cases: the sum of every cell of the volume table (ten weeks by six regions), within 0.5%.\n"
        "  (2) worst_region and worst_region_pct: the region with the largest percentage "
        "decline in cases from week 1 to week 8, and that percentage change, within 0.5 "
        "percentage points.\n"
        "  (3) anomaly_week and anomaly_drop: the week containing the single largest one-week "
        "decline in cases in any region, and that decline in percent, within 1.0 point.\n"
        "  (4) trend: up, down or flat for total cases, comparing the last two weeks with the "
        "first two (flat means within 3%).\n"
        "  (5) aht_region and aht_pct: the region with the largest percentage increase in "
        "average handling time from week 1 to week 8, and that increase, within 0.5 points.\n"
        "  (6) the summary must not contain a number that is not a cell value, a total, or a "
        "percentage derivable from the tables.\n"
        "  (7) the summary must name the worst volume region, the anomaly week and the "
        "handling-time region.\n"
        "A product is accepted only if (1)-(6) hold."
    )
    checklist = [
        "Every number in the summary appears in one of the tables or is a percentage you computed from them.",
        "total_cases was summed over all sixty volume cells, not estimated.",
        "The worst volume region was chosen by percentage change from week 1 to week 8, not by absolute volume.",
        "The anomaly week was found by scanning every consecutive week pair in every region.",
        "The trend compares the last two weeks with the first two weeks.",
        "aht_region was chosen from the handling-time table, which is a different table from the volume table.",
        "The summary names the worst volume region, the anomaly week and the handling-time region.",
    ]

    # ------------------------------------------------------------------ generate
    def generate(self, n: int, seed: int = 0) -> List[Instance]:
        rng = random.Random(seed * 104729 + 3)
        return [self._one(rng, i) for i in range(n)]

    def _one(self, rng: random.Random, idx: int) -> Instance:
        n_wk = 10
        regs = REGIONS[:]
        rng.shuffle(regs)
        base = {r: rng.randint(180, 1400) for r in regs}
        growth = {r: rng.uniform(-0.025, 0.045) for r in regs}
        decline = regs[0]
        growth[decline] = -rng.uniform(0.045, 0.065)
        anom_reg = regs[1]
        anom_wk = rng.randint(2, 6)

        cases: Dict[str, List[int]] = {r: [] for r in regs}
        for r in regs:
            v = base[r]
            for w in range(n_wk):
                v = v * (1 + growth[r] + rng.uniform(-0.035, 0.035))
                x = v
                if r == anom_reg and w == anom_wk:
                    x = v * 0.45
                cases[r].append(int(round(x)))

        aht_base = {r: rng.uniform(12.0, 34.0) for r in regs}
        aht_growth = {r: rng.uniform(-0.012, 0.012) for r in regs}
        aht_reg = regs[2]
        aht_growth[aht_reg] = rng.uniform(0.022, 0.034)
        aht: Dict[str, List[float]] = {r: [] for r in regs}
        for r in regs:
            v = aht_base[r]
            for w in range(n_wk):
                v = v * (1 + aht_growth[r] + rng.uniform(-0.008, 0.008))
                aht[r].append(round(v, 1))

        total = sum(sum(v) for v in cases.values())
        pct = {r: (cases[r][-1] - cases[r][0]) / cases[r][0] * 100 for r in regs}
        worst = min(pct, key=lambda r: pct[r])
        drops = [((cases[r][w] - cases[r][w - 1]) / cases[r][w - 1] * 100, r, w)
                 for r in regs for w in range(1, n_wk)]
        drop_pct, drop_reg, drop_wk = min(drops)
        first2 = sum(sum(cases[r][:2]) for r in regs)
        last2 = sum(sum(cases[r][-2:]) for r in regs)
        chg = (last2 - first2) / first2 * 100
        trend = "up" if chg > 3 else ("down" if chg < -3 else "flat")
        apct = {r: (aht[r][-1] - aht[r][0]) / aht[r][0] * 100 for r in regs}
        aht_worst = max(apct, key=lambda r: apct[r])

        truth = {
            "total_cases": total,
            "worst_region": worst,
            "worst_region_pct": round(pct[worst], 1),
            "anomaly_week": drop_wk + 1,
            "anomaly_drop": round(-drop_pct, 1),
            "anomaly_region": drop_reg,
            "trend": trend,
            "aht_region": aht_worst,
            "aht_pct": round(apct[aht_worst], 1),
            "cases": cases,
            "aht": aht,
            "pct_all": [round(v, 1) for v in pct.values()],
            "apct_all": [round(v, 1) for v in apct.values()],
            "drops_all": [round(d[0], 1) for d in drops],
        }
        text = self._render(cases, aht)
        canonical = self._canonical(truth)
        return Instance(family=self.name, iid=f"{self.name}-{idx:04d}", text=text,
                        truth=truth, n_checks=10, canonical=canonical,
                        meta={"anomaly_region": drop_reg, "decline_region": decline,
                              "aht_region": aht_reg})

    def _render(self, cases: Dict[str, List[int]], aht: Dict[str, List[float]]) -> str:
        regs = list(cases)
        out = ["OPERATIONS PACK, FY26 Q1", ""]
        hdr = "week | " + " | ".join(f"{r:>8}" for r in regs)
        out += ["TABLE 1 - cases resolved per week, by region", "", hdr, "-" * len(hdr)]
        for w in range(len(cases[regs[0]])):
            out.append(f"  {w+1}  | " + " | ".join(f"{cases[r][w]:>8d}" for r in regs))
        out += ["", "TABLE 2 - average handling time in minutes per case, by region", "",
                hdr, "-" * len(hdr)]
        for w in range(len(aht[regs[0]])):
            out.append(f"  {w+1}  | " + " | ".join(f"{aht[r][w]:>8.1f}" for r in regs))
        out += ["", "Weeks run Monday-Sunday. Cases are cases resolved, not cases opened."]
        return "\n".join(out)

    def _canonical(self, t: Dict[str, Any]) -> str:
        return json.dumps({
            "findings": {k: t[k] for k in FIELDS},
            "summary": (f"Resolved volume totalled {t['total_cases']:,} cases over the eight "
                        f"weeks. {t['worst_region']} fell {abs(t['worst_region_pct']):.1f}% "
                        f"between week 1 and week 8, the largest regional decline. Week "
                        f"{t['anomaly_week']} carries the sharpest single-week drop, "
                        f"{t['anomaly_drop']:.1f}% in {t['anomaly_region']}. The aggregate "
                        f"trend is {t['trend']}. Handling time moved the other way: "
                        f"{t['aht_region']} rose {t['aht_pct']:.1f}%, the largest increase in "
                        "the second table, which points at capacity rather than demand."),
        }, indent=1)

    # ------------------------------------------------------------------ prompts
    def producer_system(self) -> str:
        return (
            "You are an operations analyst. Read the two weekly tables and write the quarterly "
            "summary. Every quantitative claim you make must be checkable against the tables; do "
            "not estimate, and do not introduce numbers from outside them.\n\n"
            "Compute exactly these findings:\n"
            "  total_cases       sum of all cells in table 1.\n"
            "  worst_region      region with the largest percentage decline in cases from week 1 "
            "to week 8.\n"
            "  worst_region_pct  that percentage change (negative).\n"
            "  anomaly_week      the week (1-8) containing the single largest one-week decline in "
            "cases in any region.\n"
            "  anomaly_drop      that decline in percent (positive number).\n"
            "  trend             up, down or flat for total cases, comparing weeks 7-8 with "
            "weeks 1-2; flat means within 3%.\n"
            "  aht_region        region with the largest percentage increase in average handling "
            "time (table 2) from week 1 to week 8.\n"
            "  aht_pct           that increase in percent (positive number).\n\n"
            "Respond with JSON only:\n"
            '{"findings":{"total_cases":0,"worst_region":"...","worst_region_pct":0.0,'
            '"anomaly_week":0,"anomaly_drop":0.0,"trend":"up|down|flat","aht_region":"...",'
            '"aht_pct":0.0},"summary":"<five to seven sentences>"}'
        )

    def verifier_system(self) -> str:
        return (
            "You are an independent reviewer of an operations summary. You did not write it. "
            "Recompute every figure from the two tables yourself - the volume total, every "
            "regional percentage change in both tables, and every consecutive-week change - then "
            "check the summary sentence by sentence. A number in the summary that you cannot "
            "reproduce from the tables is a critical defect even if it looks plausible.\n\n"
            "Respond with JSON only:\n"
            '{"accepted":true|false,"checks":<int>,"defects":[{"field":"total_cases|worst_region|worst_region_pct|'
            'anomaly_week|anomaly_drop|trend|aht_region|aht_pct|unsupported_number|'
            'missing_reference","severity":"critical|major|minor","expected":"<your value>"}]}\n'
            "Set accepted to false if any figure is wrong or any summary number cannot be "
            "reproduced from the tables."
        )

    # ------------------------------------------------------------------ grading
    def census(self, inst: Instance, product: Optional[dict]) -> List[Defect]:
        if not product:
            return [Defect("unparseable_product", "critical", "no JSON product returned")]
        f = product.get("findings") if isinstance(product.get("findings"), dict) else {}
        t = inst.truth
        defects: List[Defect] = []

        tot = as_float(f.get("total_cases"))
        if tot is None or abs(tot - t["total_cases"]) > 0.005 * t["total_cases"]:
            defects.append(Defect("wrong_total", "major", f"{tot} vs {t['total_cases']}"))
        wr = str(f.get("worst_region", "")).strip()
        if wr.lower() != str(t["worst_region"]).lower():
            defects.append(Defect("wrong_worst_region", "major",
                                  f"{wr!r} vs {t['worst_region']!r}"))
        wp = as_float(f.get("worst_region_pct"))
        if wp is None or abs(wp - t["worst_region_pct"]) > 0.5:
            defects.append(Defect("wrong_worst_pct", "major", f"{wp} vs {t['worst_region_pct']}"))
        aw = as_float(f.get("anomaly_week"))
        if aw is None or int(round(aw)) != t["anomaly_week"]:
            defects.append(Defect("wrong_anomaly_week", "major", f"{aw} vs {t['anomaly_week']}"))
        ad = as_float(f.get("anomaly_drop"))
        if ad is None or abs(ad - t["anomaly_drop"]) > 1.0:
            defects.append(Defect("wrong_anomaly_drop", "major", f"{ad} vs {t['anomaly_drop']}"))
        tr = str(f.get("trend", "")).strip().lower()
        if tr != t["trend"]:
            defects.append(Defect("wrong_trend", "minor", f"{tr!r} vs {t['trend']!r}"))
        ar = str(f.get("aht_region", "")).strip()
        if ar.lower() != str(t["aht_region"]).lower():
            defects.append(Defect("wrong_aht_region", "major", f"{ar!r} vs {t['aht_region']!r}"))
        ap = as_float(f.get("aht_pct"))
        if ap is None or abs(ap - t["aht_pct"]) > 0.5:
            defects.append(Defect("wrong_aht_pct", "major", f"{ap} vs {t['aht_pct']}"))

        summary = str(product.get("summary", ""))
        allowed = self._allowed_numbers(t)
        for tok in re.findall(r"\d[\d,]*\.?\d*%?", summary):
            raw = tok.rstrip("%").replace(",", "")
            try:
                v = float(raw)
            except ValueError:
                continue
            if len(raw.replace(".", "")) < 2:
                continue                                    # single digits: week/region counts
            if not any(abs(v - a) <= max(0.6, 0.02 * abs(a)) for a in allowed):
                defects.append(Defect("unsupported_number", "critical", f"'{tok}' in summary"))
        low = summary.lower()
        if str(t["worst_region"]).lower() not in low:
            defects.append(Defect("missing_region_mention", "minor", str(t["worst_region"])))
        if not re.search(r"\bweek\s*%d\b" % t["anomaly_week"], low):
            defects.append(Defect("missing_week_mention", "minor", str(t["anomaly_week"])))
        if str(t["aht_region"]).lower() not in low:
            defects.append(Defect("missing_aht_mention", "minor", str(t["aht_region"])))
        return defects

    @staticmethod
    def _allowed_numbers(t: Dict[str, Any]) -> List[float]:
        out: List[float] = []
        out += [x for v in t["cases"].values() for x in v]
        out += [x for v in t["aht"].values() for x in v]
        out += [t["total_cases"], t["worst_region_pct"], abs(t["worst_region_pct"]),
                t["anomaly_drop"], t["anomaly_week"], t["aht_pct"], abs(t["aht_pct"])]
        out += [abs(x) for x in t.get("pct_all", [])] + list(t.get("pct_all", []))
        out += [abs(x) for x in t.get("apct_all", [])] + list(t.get("apct_all", []))
        out += [abs(x) for x in t.get("drops_all", [])] + list(t.get("drops_all", []))
        out += [round(x, 1) for x in out]
        out += [round(x) for x in out]
        out += [float(i) for i in range(1, 13)]
        return out
