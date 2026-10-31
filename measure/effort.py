"""Effort accounting.

All three components of the debt ratio are expressed in one unit: **tokens of
text processed**, i.e. the token volume a procedure has to read plus write in
order to do its job.  Two of the three are measured exactly by the local model
server; the third (the gross production saving) is computed from the character
volume of the work and converted with a ratio calibrated on the run's own logs.

    S  gross production saving   = tokens to read the raw inputs
                                   + tokens to write the full correct deliverable
    V  verification cost         = tokens processed by the independent verifier,
                                   summed over every pass
    R  rework cost               = tokens processed by every repair call

The unit is a *volume* proxy for effort, not a clock.  Wall-clock is recorded
alongside and used as a robustness check (``unit="time"``).  Nothing in this
module converts tokens into hours or money; that is left to the sensitivity
analysis in the paper, which states its conversion factors explicitly.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def calibrate(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Chars-per-token ratios, estimated from this run's own traffic.

    ``rho_in`` is used to price reading, ``rho_out`` to price writing.  Both are
    ratios of character volume to the server's own token counts, so they inherit
    the tokenizer of the models actually used.
    """
    in_chars = in_tok = out_chars = out_tok = 0
    for r in records:
        in_chars += r.get("in_chars", 0)
        in_tok += r.get("prompt_tokens", 0)
        out_chars += r.get("out_chars", 0)
        out_tok += r.get("completion_tokens", 0)
    rho_in = in_chars / in_tok if in_tok else 3.7
    rho_out = out_chars / out_tok if out_tok else 3.7
    return {"rho_in": round(rho_in, 4), "rho_out": round(rho_out, 4),
            "in_chars": in_chars, "in_tokens": in_tok,
            "out_chars": out_chars, "out_tokens": out_tok}


def saving_tokens(inst_chars: int, canon_chars: int, rho: Dict[str, float]) -> float:
    return inst_chars / max(rho["rho_in"], 1e-6) + canon_chars / max(rho["rho_out"], 1e-6)


def components(r: Dict[str, Any], rho: Dict[str, float]) -> Dict[str, float]:
    """S, V, R, and the production cost P (kept out of D by definition)."""
    S = saving_tokens(r["inst_chars"], r["canon_chars"], rho)
    V = float(r.get("ver_tokens", 0))
    R = float(r.get("rework_tokens", 0))
    P = float(r.get("prod_tokens", 0))
    return {"S": S, "V": V, "R": R, "P": P}


def debt_ratio(r: Dict[str, Any], rho: Dict[str, float]) -> float:
    c = components(r, rho)
    return (c["V"] + c["R"]) / max(c["S"], 1e-9)


def total_effort(r: Dict[str, Any]) -> float:
    """Everything the deployment spends on one accepted product, in tokens."""
    return float(r.get("prod_tokens", 0) + r.get("ver_tokens", 0) + r.get("rework_tokens", 0))
