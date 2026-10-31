"""Debt-ratio estimation, rank correlation, and the guardrail comparison.

Everything here is a pure function of ``results/*.jsonl``.  No number in the paper
is typed by hand; ``make_figs.py`` and ``run_all.py summarize`` both go through
these functions.

Definitions (paper section 3):

    D_f = (V_f + R_f) / S_f          ratio of sums over the family's instances
    net = S_f (1 - D_f)              accounting identity; D >= 1 destroys value

``D`` is estimated as a ratio of sums, not a mean of ratios: the accounting
quantity is total cost per unit of total work displaced.  Both are reported.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import effort


# --------------------------------------------------------------------- helpers
def _ratio(recs: Sequence[Dict[str, Any]], rho: Dict[str, float]) -> float:
    S = V = R = 0.0
    for r in recs:
        c = effort.components(r, rho)
        S += c["S"]; V += c["V"]; R += c["R"]
    return (V + R) / max(S, 1e-9)


def _mean_ratio(recs: Sequence[Dict[str, Any]], rho: Dict[str, float]) -> float:
    if not recs:
        return float("nan")
    return sum(effort.debt_ratio(r, rho) for r in recs) / len(recs)


def bootstrap_ratio(recs: Sequence[Dict[str, Any]], rho: Dict[str, float],
                    n_boot: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    obs = _ratio(recs, rho)
    n = len(recs)
    if n < 2:
        return obs, obs, obs
    draws = []
    for _ in range(n_boot):
        sample = [recs[rng.randrange(n)] for _ in range(n)]
        draws.append(_ratio(sample, rho))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return obs, lo, hi


def family_table(records: Sequence[Dict[str, Any]], rho: Dict[str, float],
                 arm: str = "baseline", model: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    fams = sorted({r["family"] for r in records})
    for f in fams:
        recs = [r for r in records if r["family"] == f and r["arm"] == arm
                and (model is None or r["model"] == model)]
        if not recs:
            continue
        S = sum(effort.components(r, rho)["S"] for r in recs)
        V = sum(effort.components(r, rho)["V"] for r in recs)
        R = sum(effort.components(r, rho)["R"] for r in recs)
        P = sum(effort.components(r, rho)["P"] for r in recs)
        obs, lo, hi = bootstrap_ratio(recs, rho)
        out.append({
            "family": f, "n": len(recs), "S": S, "V": V, "R": R, "P": P,
            "D": (V + R) / max(S, 1e-9), "D_lo": lo, "D_hi": hi,
            "D_mean": _mean_ratio(recs, rho),
            "D_full": (P + V + R) / max(S, 1e-9),
            "V_over_S": V / max(S, 1e-9), "R_over_S": R / max(S, 1e-9),
            "V_share": V / max(V + R, 1e-9),
            "pass1_accept": sum(1 for r in recs if r["census_first"]["blocking"] == 0) / len(recs),
            "final_accept": sum(1 for r in recs if r["census_final"]["blocking"] == 0) / len(recs),
            "verifier_accept": sum(1 for r in recs if r["accepted_by_verifier"]) / len(recs),
            "rework_rate": sum(1 for r in recs if r["n_rework_rounds"] > 0) / len(recs),
            "mean_rounds": sum(r["n_rework_rounds"] for r in recs) / len(recs),
            "effort_per_product": sum(effort.total_effort(r) for r in recs) / len(recs),
        })
    return out


# ------------------------------------------------------------- rank correlation
def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(list(x)), rank(list(y))
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def permutation_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Exact two-sided permutation p-value (n! permutations; n=4 -> 24)."""
    obs = abs(spearman(x, y))
    n = len(x)
    if n < 3 or obs != obs:          # NaN or too few points to permute
        return 1.0
    if n > 8:
        rng = random.Random(0)
        hits = 0
        trials = 20000
        for _ in range(trials):
            perm = list(y)
            rng.shuffle(perm)
            hits += abs(spearman(x, perm)) >= obs - 1e-12
        return hits / trials
    hits = total = 0
    for perm in itertools.permutations(range(n)):
        total += 1
        if abs(spearman(x, [y[i] for i in perm])) >= obs - 1e-12:
            hits += 1
    return hits / total


# ---------------------------------------------------------------- guardrails
def guardrail_table(records: Sequence[Dict[str, Any]], rho: Dict[str, float],
                    n_boot: int = 4000, seed: int = 0) -> List[Dict[str, Any]]:
    """Baseline vs guardrail on the *same* instances, with a paired bootstrap.

    The baseline arm has more instances than the guardrail arm, so an unmatched
    comparison would price the guardrail against a different (larger) sample.
    We restrict the baseline to the (model, iid) pairs that also appear in the
    guardrail arm, then bootstrap over those pairs to get a CI on the change in D.
    """
    out: List[Dict[str, Any]] = []
    for f in sorted({r["family"] for r in records}):
        base_all = [r for r in records if r["family"] == f and r["arm"] == "baseline"]
        guard = [r for r in records if r["family"] == f and r["arm"] == "guardrail"]
        if not base_all or not guard:
            continue
        gkeys = {(r["model"], r["iid"]) for r in guard}
        base = [r for r in base_all if (r["model"], r["iid"]) in gkeys]
        if not base:
            continue
        Db, Dg = _ratio(base, rho), _ratio(guard, rho)
        # paired bootstrap over matched (model, iid) keys
        keys = sorted(gkeys)
        bmap = {(r["model"], r["iid"]): r for r in base}
        gmap = {(r["model"], r["iid"]): r for r in guard}
        rng = random.Random(seed)
        diffs = []
        for _ in range(n_boot):
            sample = [keys[rng.randrange(len(keys))] for _ in keys]
            diffs.append(_ratio([gmap[k] for k in sample], rho)
                         - _ratio([bmap[k] for k in sample], rho))
        diffs.sort()
        lo = diffs[int(0.025 * len(diffs))]
        hi = diffs[int(0.975 * len(diffs)) - 1]
        Eb = sum(effort.total_effort(r) for r in base) / len(base)
        Eg = sum(effort.total_effort(r) for r in guard) / len(guard)
        _, blo, bhi = bootstrap_ratio(base, rho)
        _, glo, ghi = bootstrap_ratio(guard, rho)
        out.append({
            "family": f, "n_base": len(base), "n_guard": len(guard),
            "D_base": Db, "D_guard": Dg, "D_base_lo": blo, "D_base_hi": bhi,
            "D_guard_lo": glo, "D_guard_hi": ghi,
            "delta_D": Dg - Db, "delta_lo": lo, "delta_hi": hi,
            "cut_pct": 100.0 * (Db - Dg) / Db if Db else float("nan"),
            "effort_base": Eb, "effort_guard": Eg,
            "throughput_cost_pct": 100.0 * (Eg / Eb - 1.0),
            "pass1_accept_base": sum(1 for r in base if r["census_first"]["blocking"] == 0) / len(base),
            "pass1_accept_guard": sum(1 for r in guard if r["census_first"]["blocking"] == 0) / len(guard),
            "rework_base": sum(r["n_rework_rounds"] for r in base) / len(base),
            "rework_guard": sum(r["n_rework_rounds"] for r in guard) / len(guard),
        })
    return out


# ------------------------------------------------------------------- leakage
def leakage(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Products the verifier accepted that the census still rejects.

    This is the part of the debt that a weak checking process cannot see, and the
    reason the measured R is a lower bound on the rework a real organization faces.
    """
    acc = [r for r in records if r["accepted_by_verifier"]]
    if not acc:
        return {"n_accepted": 0, "leak_rate": float("nan"), "leak_critical": float("nan")}
    leak = [r for r in acc if r["census_final"]["blocking"] > 0]
    crit = [r for r in acc if r["census_final"]["counts"]["critical"] > 0]
    return {
        "n_accepted": len(acc),
        "leak_rate": len(leak) / len(acc),
        "leak_critical": len(crit) / len(acc),
        "missed_defects_mean": (sum(r["census_final"]["blocking"] for r in leak) / len(leak))
        if leak else 0.0,
    }


# -------------------------------------------------------- proxy sensitivity
def sensitivity(records: Sequence[Dict[str, Any]], rho: Dict[str, float],
                lambdas: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                arm: str = "baseline") -> List[Dict[str, Any]]:
    """Recompute D under a band of human/agent effort conversion factors.

    ``lambda`` multiplies measured agent effort before it is compared with the
    production saving.  Because S does not move, D is linear in lambda: the level
    of every debt ratio scales with the conversion factor, while the ordering
    across families does not.  That is the substantive claim the band supports.
    """
    rows: List[Dict[str, Any]] = []
    for lam in lambdas:
        for f in sorted({r["family"] for r in records}):
            recs = [r for r in records if r["family"] == f and r["arm"] == arm]
            if not recs:
                continue
            S = sum(effort.components(r, rho)["S"] for r in recs)
            V = sum(effort.components(r, rho)["V"] for r in recs)
            R = sum(effort.components(r, rho)["R"] for r in recs)
            rows.append({"lambda": lam, "family": f, "D": lam * (V + R) / max(S, 1e-9)})
    return rows


# ----------------------------------------------------- termination-rule sweep
def termination_sensitivity(records: Sequence[Dict[str, Any]], rho: Dict[str, float],
                           caps: Sequence[int] = (1, 2, 3, 5),
                           patiences: Sequence[int] = (1, 2),
                           arm: str = "baseline") -> List[Dict[str, Any]]:
    """Recompute the debt ratio under alternative loop-termination rules.

    The deployed loop stops at ``max_rounds`` (the cap) or when its stall
    detector trips (the same flagged field set twice, or ``patience`` consecutive
    rounds with no reduction in the defect count).  Because most products never
    clear the bar, the cap mainly bounds the worst case and the stall detector
    decides when to escalate to a human; both are measurement choices, not
    properties of the work.  This sweep replays the stored per-round verdicts
    (verifier tokens are exact; per-round rework (producer) tokens are taken as
    the product's mean rework cost, since the logs store the rework total but not
    its per-round split) under each (cap, patience) pair and recomputes D.  Only
    rules at or tighter than the deployed one are replayed, so no extrapolation
    beyond the recorded rounds is required.

    Rows are pooled over models and reported as a ratio of sums,
    ``D = sum(V+R)/sum(S)``, matching the reported ``D`` (so replaying the
    deployed rule ``cap=5, patience=2`` reproduces the recorded pooled ``D``
    exactly; see ``tests/test_termination_replay.py``).
    """
    acc: Dict[Tuple[str, int, int], List[float]] = {}  # (family, cap, pat) -> [sumV, sumR, sumS]
    for r in records:
        if r.get("arm") != arm:
            continue
        rounds = r.get("rounds", [])
        if not isinstance(rounds, list) or not rounds:
            continue
        n_rework = int(r.get("n_rework_rounds", 0) or 0)
        r_per = (float(r.get("rework_tokens", 0) or 0) / n_rework) if n_rework > 0 else 0.0
        S = effort.components(r, rho)["S"]
        # per-round quantities, aligned with the round index (a rework round that
        # tripped the identical-product break ships without a verifier verdict)
        v_tokens, flags_seq, accepted_seq = [], [], []
        for rd in rounds:
            v = rd.get("verifier") if isinstance(rd, dict) else None
            if v is not None:
                v_tokens.append(float(v.get("total_tokens", 0) or 0))
                defects = v.get("defects", [])
                if isinstance(defects, list):
                    fl = frozenset(
                        (str(d.get("invoice_id") or d.get("email_id") or d.get("email")
                             or d.get("field")), str(d.get("field", "")))
                        for d in defects if isinstance(d, dict))
                else:
                    fl = frozenset()
                flags_seq.append(fl)
                accepted_seq.append(bool(v.get("accepted", False)))
            else:
                v_tokens.append(0.0)
                flags_seq.append(frozenset())
                accepted_seq.append(False)

        for cap in caps:
            for pat in patiences:
                V = (v_tokens[0] if v_tokens else 0.0)
                R = 0.0
                key = (r["family"], cap, pat)
                if accepted_seq and accepted_seq[0]:
                    a = acc.setdefault(key, [0.0, 0.0, 0.0])
                    a[0] += V; a[1] += R; a[2] += S
                    continue
                # Mirror the deployed loop's per-iteration order exactly:
                #   verify round i (count V_i) -> check accepted -> check cap
                #   -> check stall -> rework to round i+1 (count R) -> continue.
                # Round 0 is verified before the loop; the loop then reworks to
                # round 1 (the first rework) if the cap allows.
                prev_flags = flags_seq[0] if flags_seq else None
                prev_n = len(prev_flags) if prev_flags is not None else None
                no_progress = 0
                if cap >= 1 and len(rounds) > 1:
                    R += r_per                      # rework round 0 -> 1
                i = 1
                while i < len(rounds):
                    V += (v_tokens[i] if i < len(v_tokens) else 0.0)   # verify round i
                    if i < len(accepted_seq) and accepted_seq[i]:
                        break
                    if i >= cap:                    # n_round == i reworks done
                        break
                    flags = flags_seq[i] if i < len(flags_seq) else frozenset()
                    if prev_flags is not None and flags and flags == prev_flags:
                        break                       # identical flagged set -> escalate
                    if prev_n is not None and len(flags) >= prev_n:
                        no_progress += 1
                        if no_progress >= pat:
                            break                   # no progress -> escalate
                    else:
                        no_progress = 0
                    prev_n = len(flags)
                    prev_flags = flags
                    if i + 1 < len(rounds):
                        R += r_per                  # rework round i -> i+1
                    i += 1
                a = acc.setdefault(key, [0.0, 0.0, 0.0])
                a[0] += V; a[1] += R; a[2] += S

    out = [{"model": "pooled", "family": fam, "cap": cap, "patience": pat,
            "D": (sv + sr) / max(ss, 1e-9), "V": sv, "R": sr, "S": ss}
           for (fam, cap, pat), (sv, sr, ss) in sorted(acc.items())]
    return out


def detection_agreement(records, arm="baseline"):
    """Product-level agreement between the verifier and the census at first pass.

    The verifier decides accept/reject on the whole product; the census decides
    whether the product has a critical/major (blocking) defect.  Matching at the
    product level is the operationally relevant test --- it is the level at which
    the accept/reject decision is made --- and it sidesteps the schema mismatch
    between the verifier (``field``) and the census (``code`` + free-text
    ``detail``), which do not map one-to-one.

    Returns pooled and per-family 2x2 counts, precision, recall, the share of
    first-pass rejections that fall on census-clean products (false rejections),
    and the share of rework tokens spent on those products.
    """
    def _blank():
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0,
                "rework_tokens": 0.0, "rework_false_tokens": 0.0, "n": 0}

    pooled = _blank()
    by_family = {}
    for r in records:
        if r.get("arm") != arm:
            continue
        rounds = r.get("rounds", [])
        v = rounds[0].get("verifier") if rounds else None
        v_rej = v is not None and not bool(v.get("accepted", False))
        c_flag = r.get("census_first", {}).get("blocking", 0) > 0
        rw = float(r.get("rework_tokens", 0) or 0)
        cell = ("TP" if v_rej and c_flag else
                "FP" if v_rej and not c_flag else
                "FN" if (not v_rej) and c_flag else "TN")
        for d in (pooled, by_family.setdefault(r["family"], _blank())):
            d[cell] += 1
            d["n"] += 1
            d["rework_tokens"] += rw
            if cell == "FP":
                d["rework_false_tokens"] += rw

    def _fin(d):
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        d["precision"] = tp / (tp + fp) if (tp + fp) else None
        d["recall"] = tp / (tp + fn) if (tp + fn) else None
        d["false_rej_share"] = fp / (tp + fp) if (tp + fp) else None
        d["rework_false_share"] = (d["rework_false_tokens"] / d["rework_tokens"]
                                   if d["rework_tokens"] else None)
        return d

    return {"pooled": _fin(pooled),
            "by_family": {k: _fin(v) for k, v in sorted(by_family.items())}}


def checker_attribution(records, rho, gold_d, arm="baseline", B=4000, seed=13):
    """Attribute the EXCESS over a one-pass gold baseline to the checker.

    A first verification pass is a structural floor, not checker error: the verifier
    must re-read the inputs once regardless (Section 5), and the gold control prices
    exactly that one-pass cost as ``D_gold = V_gold / S_gold`` for each (family,
    model). Attributing the *whole* trajectory's V+R to a false first rejection
    therefore double-counts the unavoidable floor. We instead attribute only the
    excess:

        excess_i = (V_i + R_i) - S_i * D_gold(family_i, model_i),

    floored at zero (a trajectory never costs less than one verification pass), for
    every trajectory whose *first* rejection was on a census-clean product. The
    attributable total is expressed as (i) a share of all V+R and (ii) a D-like
    ratio ``excess / total S``.

    Intervals are a 4000-draw bootstrap resampling whole products (records) with
    the given seed, matching the paper's interval convention; ``D_gold`` is a fixed
    positive-control benchmark and is *not* resampled (it carries no sampling
    variance in this analysis). ``gold_d`` maps ``(family, model) -> D_gold``.
    """
    rows = []
    for r in records:
        if r.get("arm") != arm:
            continue
        rounds = r.get("rounds", [])
        v = rounds[0].get("verifier") if rounds else None
        rej = v is not None and not bool(v.get("accepted", False))
        clean = r.get("census_first", {}).get("blocking", 0) == 0
        S = effort.components(r, rho)["S"]
        VR = float(r.get("ver_tokens", 0) or 0) + float(r.get("rework_tokens", 0) or 0)
        dg = gold_d.get((r["family"], r["model"]))
        baseline = S * dg if dg is not None else 0.0
        excess = max(0.0, VR - baseline)
        rows.append((excess, S, VR, bool(rej and clean)))

    tot_vr = sum(x[2] for x in rows)
    tot_s = sum(x[1] for x in rows)
    chk_excess = sum(x[0] for x in rows if x[3])

    rnd = random.Random(seed)
    n = len(rows)
    shares, chk_d = [], []
    for _ in range(B):
        ev = ts = tv = 0.0
        for _ in range(n):
            ex, S, VR, chk = rows[rnd.randrange(n)]
            ts += S; tv += VR
            if chk:
                ev += ex
        if tv > 0:
            shares.append(ev / tv)
        if ts > 0:
            chk_d.append(ev / ts)

    def q(a, p):
        a = sorted(a)
        return a[min(len(a) - 1, max(0, int(p * len(a))))]

    return {
        "checker_excess": chk_excess, "total_vr": tot_vr, "total_s": tot_s,
        "share": chk_excess / tot_vr if tot_vr else None,
        "share_lo": q(shares, 0.025), "share_hi": q(shares, 0.975),
        "D_check": chk_excess / tot_s if tot_s else None,
        "D_check_lo": q(chk_d, 0.025), "D_check_hi": q(chk_d, 0.975),
        "D_total": tot_vr / tot_s if tot_s else None,
    }
