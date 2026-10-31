#!/usr/bin/env python3
"""Diagnostic: one instance end-to-end, dump verifier vs census side by side."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backends.ollama_backend import OllamaBackend
from tasks import FAMILIES
from agents.producer import produce
from agents.verifier import verify
from measure.census import census

fam = sys.argv[1] if len(sys.argv) > 1 else "invoices"
model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"

inst = FAMILIES[fam].generate(1, 13)[0]
print("=" * 70)
print(f"FAMILY={fam} MODEL={model}")
print(f"n_checks={inst.n_checks} s_chars={inst.s_chars}")

b = OllamaBackend(model, num_predict=420)
first = produce(b, FAMILIES[fam], inst, "baseline")
print("-" * 70)
print("PRODUCER PRODUCT:")
print(first["text"][:2500])
print("-" * 70)

v = verify(b, FAMILIES[fam], inst, first["text"])
print(f"VERIFIER accepted={v['accepted']} n_defects={len(v['defects'])} error={v.get('error')}")
print("RAW VERIFIER TEXT:")
print(repr(v["text"][:1500]))
print("PARSED:", json.dumps(v.get("raw"), indent=2)[:1500])
print("DEFECTS:")
print(json.dumps(v["defects"], indent=2)[:3000])

cf = census(FAMILIES[fam], inst, first["text"])
print("-" * 70)
print(f"CENSUS counts={cf['counts']} blocking={cf['blocking']}")
print(json.dumps(cf["defects"], indent=2)[:3000])
print("=" * 70)
