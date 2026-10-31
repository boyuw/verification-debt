"""Rework harness: return a rejected product, repair it, re-verify, repeat.

The loop is gated by the *independent verifier*, not by ground truth, because that
is what a real deployment would do: the organization only reworks what its
checking process catches.  Ground truth is used afterwards, by the census, to
measure how much slipped through.  That gap is reported as ``leakage``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backends.ollama_backend import OllamaBackend
from tasks.base import Instance, TaskFamily

from .producer import produce
from .verifier import verify, verify_dual


def run_one(backend: OllamaBackend, fam: TaskFamily, inst: Instance, arm: str = "baseline",
            max_rounds: int = 5, dual: bool = True,
            verifier_backend: Optional[OllamaBackend] = None) -> Dict[str, Any]:
    """One work product end to end, with full effort accounting.

    ``backend`` produces; ``verifier_backend`` (defaults to ``backend``) verifies.
    Decoupling the two lets a sweep hold the verification process fixed while
    varying the producer model, isolating producer quality from verifier quality.
    """
    vb = verifier_backend or backend
    rounds: List[Dict[str, Any]] = []

    first = produce(backend, fam, inst, arm=arm)
    product_text = first["text"]
    rounds.append({"round": 0, "kind": "produce", "producer": first})

    prod_tokens = first["total_tokens"]
    rework_tokens = 0
    ver_tokens = 0
    prompt_tokens = first["prompt_tokens"]
    completion_tokens = first["completion_tokens"]
    in_chars = first["system_chars"] + first["user_chars"]
    out_chars = first["out_chars"]

    n_round = 0
    stalled = False
    prev_flags = None
    prev_n = None
    no_progress = 0
    while True:
        v = (verify_dual(vb, fam, inst, product_text)
             if (arm == "guardrail" and dual) else verify(vb, fam, inst, product_text))
        ver_tokens += v["total_tokens"]
        prompt_tokens += v["prompt_tokens"]
        completion_tokens += v["completion_tokens"]
        in_chars += v["in_chars"]
        out_chars += v["out_chars"]
        rounds[-1]["verifier"] = {
            "accepted": v["accepted"], "defects": v["defects"],
            "verdict_error": v.get("verdict_error", False),
            "total_tokens": v["total_tokens"], "n_passes": v.get("n_passes", 1),
            "latency_s": v["latency_s"],
        }
        # A verifier that cannot return a readable verdict is an escalation, not a
        # rework loop: stop and record it as stalled so R is not inflated.
        if v.get("verdict_error"):
            stalled = True
            break
        if v["accepted"] or n_round >= max_rounds:
            break
        # stall detector: the loop ends when it stops making progress, because a
        # deployment would escalate to a human at that point.  Two triggers: an
        # identical product, or the same set of flagged fields twice running.
        flags = frozenset(
            (str(d.get("invoice_id") or d.get("email_id") or d.get("email") or d.get("field")),
             str(d.get("field", "")))
            for d in v["defects"] if isinstance(d, dict)
        )
        if flags and flags == prev_flags:
            stalled = True
            break
        if prev_n is not None and len(flags) >= prev_n:
            no_progress += 1
            if no_progress >= 2:
                stalled = True
                break
        else:
            no_progress = 0
        prev_n = len(flags)
        prev_flags = flags
        n_round += 1
        rw = produce_rework(backend, fam, inst, product_text, v, arm=arm)
        rework_tokens += rw["total_tokens"]
        prompt_tokens += rw["prompt_tokens"]
        completion_tokens += rw["completion_tokens"]
        in_chars += rw["system_chars"] + rw["user_chars"]
        out_chars += rw["out_chars"]
        if rw["text"].strip() == product_text.strip():
            stalled = True
            rounds.append({"round": n_round, "kind": "rework", "producer": rw})
            break
        product_text = rw["text"]
        rounds.append({"round": n_round, "kind": "rework", "producer": rw})

    return {
        "product_text": product_text,
        "rounds": rounds,
        "n_rework_rounds": n_round,
        "stalled": stalled,
        "accepted_by_verifier": bool(rounds[-1].get("verifier", {}).get("accepted", False)),
        "prod_tokens": prod_tokens,
        "rework_tokens": rework_tokens,
        "ver_tokens": ver_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "in_chars": in_chars,
        "out_chars": out_chars,
        "wall_s": sum(r["producer"]["latency_s"] for r in rounds)
                  + sum((r.get("verifier") or {}).get("latency_s", 0.0) for r in rounds),
    }


def produce_rework(backend: OllamaBackend, fam: TaskFamily, inst: Instance, product_text: str,
                   verdict: Dict[str, Any], arm: str = "baseline") -> Dict[str, Any]:
    system = fam.producer_system() if arm == "baseline" else fam.guarded_system()
    user = fam.rework_user(inst, product_text, {"defects": verdict.get("defects", [])})
    res = backend.chat(system=system, user=user, num_predict=1536, json_mode=True)
    return {
        "text": res.text,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        "latency_s": res.latency_s,
        "error": res.error,
        "system_chars": len(system),
        "user_chars": len(user),
        "out_chars": len(res.text),
    }
