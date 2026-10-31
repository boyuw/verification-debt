#!/usr/bin/env python3
"""Positive control: feed the planted GOLD deliverable through verifier + census.

If D cannot fall below 1 even for a perfect product, then B1 ("some D >= 1") is
guaranteed by the accounting (the verifier must re-read the inputs, which is S's
own input term) rather than an empirical finding. Also yields the verifier's
false-reject rate on gold, i.e. its precision on the clean side.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES, ORDER
from agents.verifier import verify
from measure.census import census
from measure import effort

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:14b-instruct"
b = OllamaBackend(MODEL, temperature=0.0, seed=13, num_predict=900, num_ctx=4096, top_p=1.0)

# calibrate rho on the existing logs (same as the paper)
recs = []
for fn in sorted(os.listdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw"))):
    if fn.endswith(".jsonl"):
        recs += [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw", fn))]
rho = effort.calibrate(recs)

print(f"model={MODEL} n={N}/family rho_in={rho['rho_in']:.2f} rho_out={rho['rho_out']:.2f}")
out_rows = {}
for fam in ORDER:
    F = FAMILIES[fam]
    insts = F.generate(N, 13)
    rows = []
    for inst in insts:
        cf = census(F, inst, inst.canonical)
        v = verify(b, F, inst, inst.canonical)
        S = effort.saving_tokens(len(inst.text), len(inst.canonical), rho)
        rows.append({"census_block": cf["blocking"], "ver_accept": v["accepted"],
                     "ver_defects": len(v["defects"]), "V": v["total_tokens"], "S": S})
    n = len(rows)
    cb = sum(1 for r in rows if r["census_block"] > 0)
    va = sum(1 for r in rows if r["ver_accept"])
    V = sum(r["V"] for r in rows); S = sum(r["S"] for r in rows)
    Dg = V / S
    out_rows[fam] = {"n": n, "gold_census_blocking": cb, "verifier_accept_gold": va,
                     "V_per_product": V / n, "S_per_product": S / n, "D_gold": Dg,
                     "false_reject_rate": 1 - va / n}
    print(f"{fam:10s} gold_census_blocking={cb}/{n}  verifier_accept_gold={va}/{n}  "
          f"V/prod={V/n:.0f} S/prod={S/n:.0f}  D_gold={Dg:.2f}  (false-reject rate={1-va/n:.2f})")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
slug = MODEL.replace(":", "-").replace(".", "")
path = os.path.join(HERE, "results", f"gold_control__{slug}.json")
with open(path, "w") as fh:
    json.dump({"model": MODEL, "n_per_family": N, "rho": rho, "families": out_rows}, fh, indent=1)
print("wrote", path)

