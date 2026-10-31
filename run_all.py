#!/usr/bin/env python3
"""Driver for the verification-debt experiments.

    python run_all.py pilot                     # smoke test, 2 instances/family
    python run_all.py sweep --n 150 --workers 3 # the pre-registered runs
    python run_all.py summarize                 # results/*.jsonl -> results/summary.json

Everything is local: Ollama on http://localhost:11434, no keys, no paid endpoints.
Every artifact is appended to JSONL with seed, model, decoding config and token
and wall-clock accounting, so a run can be audited and resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backends.ollama_backend import OllamaBackend          # noqa: E402
from tasks import FAMILIES, ORDER, verifiability           # noqa: E402
from agents.rework import run_one                          # noqa: E402
from measure import census as census_mod                   # noqa: E402
from measure import debt, effort                           # noqa: E402

RESULTS = os.path.join(HERE, "results")
RAW = os.path.join(RESULTS, "raw")
MODELS = ["qwen2.5:14b-instruct", "llama3.1:8b"]
PRIMARY = MODELS[0]
SEED = 13
N_DEFAULT = 150
N_GUARD = 40
MAX_ROUNDS = 5

_LOCK = threading.Lock()


def slug(model: str) -> str:
    return model.replace(":", "-").replace(".", "")


def build_instances(family: str, n: int, seed: int = SEED) -> List[Any]:
    return FAMILIES[family].generate(n, seed)


def record_of(fam_name: str, inst: Any, model: str, arm: str, out: Dict[str, Any],
              seed: int = SEED) -> Dict[str, Any]:
    fam = FAMILIES[fam_name]
    first = census_mod.census(fam, inst, out["rounds"][0]["producer"]["text"])
    final = census_mod.census(fam, inst, out["product_text"])
    return {
        "run_id": f"{fam_name}|{model}|{arm}|{inst.iid}",
        "family": fam_name,
        "iid": inst.iid,
        "model": model,
        "arm": arm,
        "seed": seed,
        "inst_chars": len(inst.text),
        "canon_chars": len(inst.canonical),
        "n_checks": inst.n_checks,
        "prod_tokens": out["prod_tokens"],
        "rework_tokens": out["rework_tokens"],
        "ver_tokens": out["ver_tokens"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        "in_chars": out["in_chars"],
        "out_chars": out["out_chars"],
        "n_rework_rounds": out["n_rework_rounds"],
        "stalled": out.get("stalled", False),
        "n_producer_calls": len(out["rounds"]),
        "accepted_by_verifier": out["accepted_by_verifier"],
        "wall_s": out["wall_s"],
        "census_first": first,
        "census_final": final,
        "product_first": out["rounds"][0]["producer"]["text"][:4000],
        "product_final": out["product_text"][:4000],
        "rounds": [{"round": r["round"], "kind": r["kind"],
                    "verifier": r.get("verifier", {})} for r in out["rounds"]],
        "config": {"temperature": 0.0, "seed": SEED, "top_p": 1.0,
                   "num_predict_producer": 1536, "num_predict_verifier": 900,
                   "verifier_model": "qwen2.5:14b-instruct (fixed)", "num_ctx": 4096},
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run_task(family: str, model: str, arm: str, inst: Any, max_rounds: int) -> Dict[str, Any]:
    # Producer and verifier are the same model (self-verification), as in the
    # original protocol. The truncation / parse bug that previously made the weak
    # model's verifier silently reject every product (inflating R) is fixed in
    # tasks/base.py:parse_verdict and the verifier token budget; verifier
    # reliability is measured and reported (see README, section 1.7, D-VERIF).
    backend = OllamaBackend(model=model, temperature=0.0, seed=SEED, num_predict=1536,
                            num_ctx=4096, top_p=1.0)
    out = run_one(backend, FAMILIES[family], inst, arm=arm, max_rounds=max_rounds,
                  verifier_backend=backend)
    return record_of(family, inst, model, arm, out)


def _existing(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["run_id"])
                except Exception:
                    continue
    return ids


def sweep(families: List[str], models: List[str], arms: List[str], n: int, n_guard: int,
          workers: int, max_rounds: int) -> None:
    os.makedirs(RAW, exist_ok=True)
    for arm in arms:
        for model in models:
            path = os.path.join(RAW, f"{arm}__{slug(model)}.jsonl")
            done = _existing(path)
            jobs = []
            for fam_name in families:
                nn = n_guard if arm == "guardrail" else n
                for inst in build_instances(fam_name, nn):
                    rid = f"{fam_name}|{model}|{arm}|{inst.iid}"
                    if rid in done:
                        continue
                    jobs.append((fam_name, inst))
            if not jobs:
                print(f"[skip] {arm}/{model}: complete ({len(done)} records)")
                continue
            print(f"[run ] {arm}/{model}: {len(jobs)} products, {workers} workers", flush=True)
            t0 = time.time()
            with open(path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_task, f, model, arm, inst, max_rounds): (f, inst)
                        for f, inst in jobs}
                for i, fut in enumerate(as_completed(futs), 1):
                    f, inst = futs[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:                    # keep the sweep alive
                        print(f"  !! {f}/{inst.iid}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    with _LOCK:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                    if i % 25 == 0 or i == len(jobs):
                        el = time.time() - t0
                        print(f"  {i}/{len(jobs)}  {el:6.0f}s  "
                              f"({el / i:.1f}s/product)", flush=True)


def load_records() -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if not os.path.isdir(RAW):
        return recs
    for fn in sorted(os.listdir(RAW)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(RAW, fn)) as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    return recs


def load_gold_d() -> Dict[Tuple[str, str], float]:
    """(family, model) -> D_gold from the positive-control (gold) runs.

    Used by ``checker_attribution`` as the fixed one-pass structural-floor
    benchmark: a single verification pass already costs ~D_gold * S, which is not
    checker error and must be subtracted before attributing cost to a false
    rejection.
    """
    golds: Dict[Tuple[str, str], float] = {}
    if not os.path.isdir(RES):
        return golds
    for fn in sorted(os.listdir(RES)):
        if not (fn.startswith("gold_control__") and fn.endswith(".json")):
            continue
        with open(os.path.join(RES, fn)) as fh:
            g = json.load(fh)
        for fam, row in g.get("families", {}).items():
            golds[(fam, g["model"])] = float(row["D_gold"])
    return golds


def summarize() -> None:
    recs = load_records()
    if not recs:
        print("no records")
        return
    rho = effort.calibrate(recs)
    # actual per-cell sample sizes (per family x model x arm), for traceability
    from collections import Counter
    cnt = Counter((r["family"], r["model"], r["arm"]) for r in recs)
    n_per_cell: Dict[str, int] = {}
    for arm in ("baseline", "guardrail"):
        vals = [v for (_f, _m, a), v in cnt.items() if a == arm]
        n_per_cell[arm] = min(vals) if vals else 0
    out: Dict[str, Any] = {
        "n_records": len(recs),
        "n_per_cell": n_per_cell,
        "n_checks": {f: next((r["n_checks"] for r in recs if r["family"] == f), 0)
                     for f in ORDER},
        "calibration": rho,
        # how much of the checker's effort is re-reading (prompt) vs emitting (completion)
        "token_split": {},
        # verifier recall on the dirty side: census-blocked products it rejected
        "verifier_recall": {},
        "verifiability": {f: verifiability(FAMILIES[f]) for f in ORDER},
        "vis_subscores": {f: FAMILIES[f].vis for f in ORDER},
        "by_model": {}, "guardrail": {}, "leakage": {}, "sensitivity": [],
    }

    for model in MODELS:
        tab = debt.family_table(recs, rho, arm="baseline", model=model)
        if not tab:
            continue
        out["by_model"][model] = tab
        vis = [verifiability(FAMILIES[t["family"]]) for t in tab]
        Ds = [t["D"] for t in tab]
        out["by_model"][model + "__rank"] = {
            "spearman": debt.spearman(vis, Ds),
            "p_perm": debt.permutation_p(vis, Ds),
            "vis": vis, "D": Ds,
            "families": [t["family"] for t in tab],
        }
        out["leakage"][model] = debt.leakage(
            [r for r in recs if r["model"] == model and r["arm"] == "baseline"])
        out["guardrail"][model] = debt.guardrail_table(
            [r for r in recs if r["model"] == model], rho)
        out["sensitivity"] += [dict(row, model=model)
                               for row in debt.sensitivity(
                                   [r for r in recs if r["model"] == model], rho, arm="baseline")]

    # loop-termination sensitivity (cap / stall patience): a measurement choice,
    # replayed offline from the stored per-round verdicts
    out["termination_sensitivity"] = debt.termination_sensitivity(recs, rho)

    # verifier-vs-census product-level agreement at first pass: quantifies the
    # false-rejection component of the debt (no GPU; derived from stored logs)
    out["detection_agreement"] = debt.detection_agreement(recs)

    # checker-attributable EXCESS over a one-pass gold baseline: trajectories whose
    # first rejection was false, cost beyond the structural floor (Section 5)
    gold_d = load_gold_d()
    out["checker_attribution"] = debt.checker_attribution(recs, rho, gold_d)

    # pooled across models (primary analysis)
    pooled = debt.family_table(recs, rho, arm="baseline", model=None)
    out["pooled"] = pooled
    out["pooled_rank"] = {
        "spearman": debt.spearman([verifiability(FAMILIES[t["family"]]) for t in pooled],
                                  [t["D"] for t in pooled]),
        "p_perm": debt.permutation_p([verifiability(FAMILIES[t["family"]]) for t in pooled],
                                     [t["D"] for t in pooled]),
    }
    out["pooled_guardrail"] = debt.guardrail_table(recs, rho)
    out["pooled_leakage"] = debt.leakage([r for r in recs if r["arm"] == "baseline"])

    # how much of the checker's effort is re-reading (prompt) vs emitting (completion)?
    base = [r for r in recs if r["arm"] == "baseline"]
    out["token_split"] = {}
    for f in ORDER:
        rows = [r for r in base if r["family"] == f]
        pt = sum(r["prompt_tokens"] for r in rows)
        ct = sum(r["completion_tokens"] for r in rows)
        out["token_split"][f] = pt / (pt + ct) if (pt + ct) else 0.0
    # verifier recall on the dirty side: census-blocked products the verifier rejected
    blocked = [r for r in base if r["census_first"]["blocking"] > 0]
    caught = [r for r in blocked if not r["accepted_by_verifier"]]
    out["verifier_recall"] = {"n_blocked": len(blocked), "n_caught": len(caught),
                              "recall": len(caught) / len(blocked) if blocked else None}

    # pre-registered bars
    bars = {}
    worst = max(pooled, key=lambda t: t["D"]) if pooled else None
    bars["B1_any_family_D_ge_1"] = bool(worst and worst["D"] >= 1.0)
    bars["B1_worst_family"] = worst["family"] if worst else None
    bars["B1_worst_D"] = worst["D"] if worst else None
    rk = out["pooled_rank"]
    # hypothesis is monotone *decreasing*: D rises as verifiability falls
    bars["B2_monotone"] = bool(rk["spearman"] <= -0.8 and rk["p_perm"] <= 0.10)
    bars["B2_spearman"] = rk["spearman"]
    bars["B2_p_perm"] = rk["p_perm"]
    g = next((r for r in out["pooled_guardrail"]
              if worst and r["family"] == worst["family"]), None)
    if g:
        bars["B3_cut_pct"] = g["cut_pct"]
        bars["B3_throughput_cost_pct"] = g["throughput_cost_pct"]
        bars["B3_met"] = bool(g["cut_pct"] >= 30.0 and g["throughput_cost_pct"] <= 15.0)
    out["bars"] = bars

    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # flat CSV for the paper's tables
    with open(os.path.join(RESULTS, "families.csv"), "w") as fh:
        cols = ["family", "n", "S", "V", "R", "P", "D", "D_lo", "D_hi", "D_mean",
                "D_full", "V_over_S", "R_over_S", "V_share", "pass1_accept",
                "final_accept", "verifier_accept", "rework_rate", "mean_rounds",
                "effort_per_product"]
        fh.write("model," + ",".join(cols) + "\n")
        for model, tab in out["by_model"].items():
            if model.endswith("__rank"):
                continue
            for t in tab:
                fh.write(model + "," + ",".join(
                    (f"{t[c]:.6f}" if isinstance(t[c], float) else str(t[c])) for c in cols) + "\n")
    print(json.dumps(bars, indent=1))
    print(f"wrote {RESULTS}/summary.json")


def pilot() -> None:
    ok = True
    for model in MODELS:
        b = OllamaBackend(model=model)
        print(model, "reachable:", b.health(), "| models:", b.available_models()[:4])
    for fam_name in ORDER:
        insts = build_instances(fam_name, 2)
        print(f"\n=== {fam_name}  vis={verifiability(FAMILIES[fam_name])}  "
              f"inst_chars={len(insts[0].text)} canon_chars={len(insts[0].canonical)} "
              f"checks={insts[0].n_checks}")
        for inst in insts[:1]:
            t0 = time.time()
            rec = run_task(fam_name, PRIMARY, "baseline", inst, MAX_ROUNDS)
            print(f"  {rec['run_id']}  {time.time()-t0:.1f}s  "
                  f"prod={rec['prod_tokens']} ver={rec['ver_tokens']} rw={rec['rework_tokens']}  "
                  f"rounds={rec['n_rework_rounds']}  verifier_accept={rec['accepted_by_verifier']}  "
                  f"first_defects={rec['census_first']['counts']}  final={rec['census_final']['counts']}")
            if rec["census_first"]["counts"]["critical"] == 0 and rec["census_first"]["n_defects"] == 0:
                print("    (clean first pass)")
            ok &= rec["census_final"]["parse_ok"]
    print("\npilot parse_ok:", ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "sweep", "summarize"])
    ap.add_argument("--families", default=",".join(ORDER))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--arms", default="baseline,guardrail")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--n-guard", type=int, default=N_GUARD)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    a = ap.parse_args()
    if a.cmd == "pilot":
        pilot()
    elif a.cmd == "summarize":
        summarize()
    else:
        sweep(a.families.split(","), a.models.split(","), a.arms.split(","),
              a.n, a.n_guard, a.workers, a.max_rounds)


if __name__ == "__main__":
    main()
