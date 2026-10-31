"""Local-only Ollama backend.

No API keys, no paid endpoints.  Everything goes to ``http://localhost:11434``
and the host is checked to be a loopback address before the first request, so
the released code cannot accidentally call a hosted service.

The backend owns exactly three things:
  * transport (HTTP, retries, loopback assertion),
  * determinism knobs (temperature, seed, top_p, num_predict, num_ctx),
  * bookkeeping (wall-clock latency, Ollama's own token counters, raw text).

Prompt construction lives in ``agents/``; grading lives in ``measure/``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_HOST = "http://localhost:11434"
LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class OllamaError(RuntimeError):
    pass


def assert_local(host: str) -> str:
    """Refuse to run against anything that is not a loopback address."""
    parsed = urllib.parse.urlparse(host)
    if parsed.scheme not in ("http", "https"):
        raise OllamaError(f"unsupported scheme: {parsed.scheme!r}")
    if parsed.hostname not in LOOPBACK:
        raise OllamaError(
            f"refusing non-loopback host {parsed.hostname!r}; this repo is local-only"
        )
    return host.rstrip("/")


@dataclass
class ChatResult:
    text: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    attempts: int = 1
    error: Optional[str] = None
    eval_duration_s: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "attempts": self.attempts,
            "error": self.error,
            "eval_duration_s": self.eval_duration_s,
        }


class OllamaBackend:
    """Thin wrapper around ``POST /api/chat`` with retries and determinism."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int = 320,
        num_ctx: int = 4096,
        top_p: float = 1.0,
        repeat_penalty: float = 1.05,
        timeout_s: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.host = assert_local(host)
        self.model = model
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.num_predict = int(num_predict)
        self.num_ctx = int(num_ctx)
        self.top_p = float(top_p)
        self.repeat_penalty = float(repeat_penalty)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.n_calls = 0
        self.total_latency_s = 0.0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------ config
    def config(self) -> Dict[str, Any]:
        """The decoding configuration, logged verbatim with every artifact."""
        return {
            "model": self.model,
            "host": self.host,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    # --------------------------------------------------------------- transport
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.host}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def available_models(self) -> List[str]:
        try:
            return [m["name"] for m in self._get("/api/tags").get("models", [])]
        except Exception:
            return []

    def health(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------- chat
    def chat(
        self,
        system: str,
        user: str,
        num_predict: Optional[int] = None,
        json_mode: bool = False,
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "top_p": self.top_p,
                "num_predict": int(num_predict or self.num_predict),
                "num_ctx": self.num_ctx,
                "repeat_penalty": self.repeat_penalty,
            },
        }
        if json_mode:
            payload["format"] = "json"
        if stop:
            payload["options"]["stop"] = stop

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self._post("/api/chat", payload)
                dt = time.time() - t0
                msg = (resp.get("message") or {}).get("content", "") or ""
                res = ChatResult(
                    text=msg,
                    latency_s=dt,
                    prompt_tokens=int(resp.get("prompt_eval_count") or 0),
                    completion_tokens=int(resp.get("eval_count") or 0),
                    model=self.model,
                    attempts=attempt,
                    eval_duration_s=float(resp.get("eval_duration") or 0) / 1e9,
                    raw={
                        "done_reason": resp.get("done_reason"),
                        "total_duration": resp.get("total_duration"),
                        "load_duration": resp.get("load_duration"),
                    },
                )
                self.n_calls += 1
                self.total_latency_s += dt
                self.total_completion_tokens += res.completion_tokens
                return res
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2.0 * attempt, 8.0))
        return ChatResult(
            text="", latency_s=0.0, model=self.model,
            attempts=self.max_retries, error=last_err,
        )
