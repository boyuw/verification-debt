#!/usr/bin/env python3
"""Quick probe: one instance per family, baseline arm, prints effort + acceptance.

Used during development only; not part of the paper's results.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES, ORDER
from agents.rework import run_one
from measure.census import census

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b-instruct"
b = OllamaBackend(model, num_predict=420)
for n in ORDER:
    inst = FAMILIES[n].generate(1, 13)[0]
    t = time.time()
    out = run_one(b, FAMILIES[n], inst, "baseline", max_rounds=5)
    c0 = census(FAMILIES[n], inst, out["rounds"][0]["producer"]["text"])
    cf = census(FAMILIES[n], inst, out["product_text"])
    print(f"{n:10s} {time.time()-t:6.1f}s rounds={out['n_rework_rounds']} "
          f"ver_acc={out['accepted_by_verifier']} P={out['prod_tokens']} V={out['ver_tokens']} "
          f"R={out['rework_tokens']} | first={c0['counts']} final={cf['counts']}", flush=True)
