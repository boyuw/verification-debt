"""Unit test: the termination-rule replay must reproduce the recorded debt ratio.

The paper reports D under the deployed loop rule (max_rounds=5, patience=2). The
termination-rule sensitivity sweeps *tighter* rules by replaying the stored
per-round verdicts. A sweep is only meaningful if, at the deployed rule, it gives
back exactly the recorded D --- otherwise the replay is measuring its own
approximation error (e.g. the per-round rework-token mean), not the effect of the
rule. This test enforces that.

Run:  python tests/test_termination_replay.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from measure import debt, effort  # noqa: E402

DEPLOYED_CAP = 5
DEPLOYED_PATIENCE = 2


def _load():
    recs = []
    for f in glob.glob(os.path.join(ROOT, "results/raw/baseline__*.jsonl")):
        with open(f) as fh:
            recs += [json.loads(line) for line in fh if line.strip()]
    return recs


def _rho():
    with open(os.path.join(ROOT, "results/summary.json")) as fh:
        cal = json.load(fh)["calibration"]
    return {"rho_in": float(cal["rho_in"]), "rho_out": float(cal["rho_out"])}


def _recorded_pooled_D(recs, rho):
    """Recorded D per family = ratio of sums, matching the reported D."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for r in recs:
        acc[r["family"]][0] += float(r["ver_tokens"]) + float(r["rework_tokens"])
        acc[r["family"]][1] += effort.components(r, rho)["S"]
    return {fam: v / s for fam, (v, s) in acc.items()}


def test_deployed_rule_reproduces_recorded_D():
    recs = _load()
    rho = _rho()
    recorded = _recorded_pooled_D(recs, rho)
    replay = {
        row["family"]: row["D"]
        for row in debt.termination_sensitivity(
            recs, rho, caps=[DEPLOYED_CAP], patiences=[DEPLOYED_PATIENCE])
    }
    for fam, d_rec in recorded.items():
        d_rep = replay[fam]
        assert abs(d_rec - d_rep) < 1e-9, (
            f"{fam}: recorded D={d_rec:.6f} but replay={d_rep:.6f} "
            f"(diff={abs(d_rec - d_rep):.2e}) --- replay bias, not a real effect")


def test_cap_sweep_is_monotone_on_worst_family():
    recs = _load()
    rho = _rho()
    rows = [r for r in debt.termination_sensitivity(
        recs, rho, caps=[1, 2, 3, 5], patiences=[DEPLOYED_PATIENCE])
        if r["family"] == "invoices"]
    rows.sort(key=lambda r: r["cap"])
    ds = [r["D"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(ds, ds[1:])), f"non-monotone: {ds}"


if __name__ == "__main__":
    test_deployed_rule_reproduces_recorded_D()
    test_cap_sweep_is_monotone_on_worst_family()
    print("OK: termination replay reproduces recorded D at the deployed rule; "
          "cap sweep monotone.")
