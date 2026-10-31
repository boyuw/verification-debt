#!/usr/bin/env python3
"""Throughput + convergence probe on a small sample.  Development only."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES, ORDER
from agents.rework import run_one
from measure.census import census

fam = sys.argv[1] if len(sys.argv) > 1 else "invoices"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
model = sys.argv[4] if len(sys.argv) > 4 else "qwen2.5:14b-instruct"

insts = FAMILIES[fam].generate(n, 13)


def one(inst):
    b = OllamaBackend(model, num_predict=420)
    t = time.time()
    out = run_one(b, FAMILIES[fam], inst, "baseline", max_rounds=5)
    cf = census(FAMILIES[fam], inst, out["product_text"])
    return (time.time() - t, out["n_rework_rounds"], out["accepted_by_verifier"],
            out["prod_tokens"], out["ver_tokens"], out["rework_tokens"],
            cf["counts"]["critical"] + cf["counts"]["major"])


t0 = time.time()
with ThreadPoolExecutor(max_workers=workers) as ex:
    res = list(ex.map(one, insts))
wall = time.time() - t0
print(f"{fam} n={n} workers={workers} wall={wall:.0f}s  per_product={wall/n:.1f}s")
print(f"  rounds={[r[1] for r in res]}")
print(f"  verifier_accept={sum(r[2] for r in res)}/{n}  "
      f"blocking_defects={[r[6] for r in res]}")
print(f"  mean tokens P={sum(r[3] for r in res)/n:.0f} V={sum(r[4] for r in res)/n:.0f} "
      f"R={sum(r[5] for r in res)/n:.0f}")
