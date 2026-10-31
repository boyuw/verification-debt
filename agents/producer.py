"""Producer arm: baseline and guarded.

The producer never sees ground truth and never sees the verifier's rubric.  In the
guarded arm it is given an acceptance checklist and a mandated dual pass; nothing
else about the task changes.
"""

from __future__ import annotations

from typing import Any, Dict

from backends.ollama_backend import OllamaBackend
from tasks.base import Instance, TaskFamily


def produce(backend: OllamaBackend, fam: TaskFamily, inst: Instance,
            arm: str = "baseline", num_predict: int = 1536) -> Dict[str, Any]:
    system = fam.producer_system() if arm == "baseline" else fam.guarded_system()
    res = backend.chat(system=system, user=inst.text, num_predict=num_predict, json_mode=True)
    return {
        "text": res.text,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        "latency_s": res.latency_s,
        "error": res.error,
        "attempts": res.attempts,
        "system_chars": len(system),
        "user_chars": len(inst.text),
        "out_chars": len(res.text),
    }
