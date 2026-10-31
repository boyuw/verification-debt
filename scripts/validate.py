#!/usr/bin/env python3
"""Validation: small-sample run to confirm verifier parses and agrees with census.

Reports per (family, model): verifier_accept rate, verdict_error rate,
census final blocking rate, mean rework rounds, and a verifier-vs-census
confusion (verifier accept but census blocks = leakage; verifier reject but
census clean = wasted rework).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES, ORDER
from agents.rework import run_one
from measure.census import census

MODELS = ["qwen2.5:14b-instruct", "llama3.1:8b"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
WORKERS = 3


def one(arg):
    fam, model, inst = arg
    backend = OllamaBackend(model=model, temperature=0.0, seed=13, num_predict=340,
                            num_ctx=4096, top_p=1.0)
    out = run_one(backend, FAMILIES[fam], inst, "baseline", max_rounds=5,
                  verifier_backend=backend)
    cf = census(FAMILIES[fam], inst, out["product_text"])
    vround = out["rounds"][-1].get("verifier", {})
    return (fam, model, out["accepted_by_verifier"], bool(vround.get("verdict_error")),
            cf["blocking"], out["n_rework_rounds"])


jobs = []
for fam in ORDER:
    for model in MODELS:
        insts = FAMILIES[fam].generate(N, 13)
        for inst in insts:
            jobs.append((fam, model, inst))

results = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    results = list(ex.map(one, jobs))
print(f"ran {len(results)} in {time.time()-t0:.0f}s\n")

for fam in ORDER:
    for model in MODELS:
        rows = [r for r in results if r[0] == fam and r[1] == model]
        if not rows:
            continue
        n = len(rows)
        acc = sum(r[2] for r in rows)
        err = sum(r[3] for r in rows)
        blk = sum(1 for r in rows if r[4] > 0)
        leak = sum(1 for r in rows if r[2] and r[4] > 0)      # accepted but census blocks
        waste = sum(1 for r in rows if (not r[2]) and r[4] == 0)  # rejected but census clean
        mr = sum(r[5] for r in rows) / n
        print(f"{fam:10s} {model:22s} n={n} verAcc={acc}/{n} verdictErr={err} "
              f"censusBlk={blk}/{n} leak={leak} waste={waste} meanRounds={mr:.2f}")
