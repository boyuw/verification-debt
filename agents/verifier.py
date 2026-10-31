"""Independent verifier arm.

Deliberately isolated from the producer: a different system prompt, no access to
the producer's reasoning, and a mandate to re-derive the answer from the raw
inputs before comparing.  In the guarded arm a *second* pass is run with the
first pass's findings withheld, and the product is delivered only if both passes
accept it.

The verifier's token count is the measured proxy for verification effort.  It is
a proxy, not a measurement of human effort; see paper section 5.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backends.ollama_backend import OllamaBackend
from tasks.base import Instance, TaskFamily


def verify(backend: OllamaBackend, fam: TaskFamily, inst: Instance, product_text: str,
           num_predict: int = 900) -> Dict[str, Any]:
    system = fam.verifier_system()
    user = fam.verifier_user(inst, product_text)
    res = backend.chat(system=system, user=user, num_predict=num_predict, json_mode=True)
    verdict = fam.parse_verdict(res.text)
    return {
        "text": res.text,
        "accepted": verdict["accepted"],
        "defects": verdict["defects"],
        "verdict_error": verdict.get("verdict_error", False),
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        "latency_s": res.latency_s,
        "error": res.error,
        "in_chars": len(system) + len(user),
        "out_chars": len(res.text),
    }


def verify_dual(backend: OllamaBackend, fam: TaskFamily, inst: Instance, product_text: str,
                num_predict: int = 900) -> Dict[str, Any]:
    """Two independent passes; both must accept.  Costs are summed."""
    passes: List[Dict[str, Any]] = [verify(backend, fam, inst, product_text, num_predict)]
    # the second pass runs regardless of the first: the mandate is dual
    # verification, not "verify again if the first pass failed"
    passes.append(verify(backend, fam, inst, product_text, num_predict))
    accepted = all(p["accepted"] for p in passes)
    defects: List[Dict[str, Any]] = []
    for p in passes:
        defects.extend(p["defects"])
    return {
        "accepted": accepted,
        "defects": defects[:12],
        "verdict_error": any(p.get("verdict_error", False) for p in passes),
        "total_tokens": sum(p["total_tokens"] for p in passes),
        "prompt_tokens": sum(p["prompt_tokens"] for p in passes),
        "completion_tokens": sum(p["completion_tokens"] for p in passes),
        "latency_s": sum(p["latency_s"] for p in passes),
        "in_chars": sum(p["in_chars"] for p in passes),
        "out_chars": sum(p["out_chars"] for p in passes),
        "n_passes": len(passes),
        "passes": [{"accepted": p["accepted"], "defects": p["defects"]} for p in passes],
        "text": passes[-1]["text"],
        "error": passes[-1]["error"],
    }
