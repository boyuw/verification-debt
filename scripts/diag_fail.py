#!/usr/bin/env python3
"""Find and print raw verifier outputs that fail to parse."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES, ORDER
from agents.producer import produce
from agents.verifier import verify
from measure.census import census

fam = sys.argv[1] if len(sys.argv) > 1 else "invoices"
model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

insts = FAMILIES[fam].generate(n, 13)
b = OllamaBackend(model, num_predict=420)
fails = 0
for inst in insts:
    first = produce(b, FAMILIES[fam], inst, "baseline")
    v = verify(b, FAMILIES[fam], inst, first["text"])
    if v.get("verdict_error"):
        fails += 1
        print("=" * 70)
        print(f"PARSE FAIL {fam} {inst.iid}  error={v.get('error')}  finish_reason=n/a")
        print("RAW VERIFIER TEXT (repr):")
        print(repr(v["text"]))
print(f"\n{fails}/{n} verifier parse failures on {fam}/{model}")
