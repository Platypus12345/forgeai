"""The single model client every factory stage runs on.

Design notes
------------
The brief says every stage runs on a *small, imperfect open model*. Three
things follow, and this module is built around them:

1. **The model will emit malformed JSON.** So `structured()` never trusts the
   model: it strips fences, extracts the first balanced JSON value, and on
   failure sends a short repair turn. Repairs are counted.
2. **The model will sometimes fail anyway.** Every structured call takes a
   `fallback` - a deterministic function producing a valid, useful result. A
   stage degrades; it does not crash. `Ledger.fallbacks` records each one and
   the final report prints the count, because a run that silently fell back to
   heuristics for half its stages is not a run you should trust.
3. **The model costs tokens and wall-clock.** Every call goes through the
   ledger, which enforces the per-run budget and lets the optimiser stop early.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import requests

from .config import BudgetConfig, ModelConfig


class BudgetExceeded(RuntimeError):
    """Raised when a run has spent its token or latency allowance."""


@dataclass
class CallRecord:
    stage: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    repaired: bool = False
    fell_back: bool = False


@dataclass
class Ledger:
    """Token/latency accounting + budget enforcement for one factory run."""

    budget: BudgetConfig
    started: float = field(default_factory=time.time)
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(c.prompt_tokens + c.output_tokens for c in self.calls)

    @property
    def seconds(self) -> float:
        return time.time() - self.started

    @property
    def repairs(self) -> int:
        return sum(1 for c in self.calls if c.repaired)

    @property
    def fallbacks(self) -> int:
        return sum(1 for c in self.calls if c.fell_back)

    def record(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    def check(self) -> None:
        if self.tokens > self.budget.max_tokens:
            raise BudgetExceeded(
                f"token budget exhausted: {self.tokens} > {self.budget.max_tokens}"
            )
        if self.seconds > self.budget.max_seconds:
            raise BudgetExceeded(
                f"latency budget exhausted: {self.seconds:.0f}s > {self.budget.max_seconds}s"
            )

    def exhausted(self) -> bool:
        try:
            self.check()
        except BudgetExceeded:
            return True
        return False

    def by_stage(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in self.calls:
            s = out.setdefault(
                c.stage,
                {"calls": 0, "tokens": 0, "seconds": 0.0, "repairs": 0, "fallbacks": 0},
            )
            s["calls"] += 1
            s["tokens"] += c.prompt_tokens + c.output_tokens
            s["seconds"] += c.seconds
            s["repairs"] += int(c.repaired)
            s["fallbacks"] += int(c.fell_back)
        for s in out.values():
            s["seconds"] = round(s["seconds"], 2)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "tokens": self.tokens,
            "seconds": round(self.seconds, 1),
            "json_repairs": self.repairs,
            "deterministic_fallbacks": self.fallbacks,
            "token_budget": self.budget.max_tokens,
            "latency_budget_s": self.budget.max_seconds,
            "per_stage": self.by_stage(),
        }


def estimate_tokens(text: str) -> int:
    """Cheap, provider-independent token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# JSON salvage - the single most important defence against a small model
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first valid JSON object/array out of a model's chatter.

    Small models wrap JSON in prose, in code fences, emit trailing commas, and
    truncate. This tries progressively dirtier repairs and raises ValueError
    only when nothing works.
    """
    if not text:
        raise ValueError("empty model output")
    candidates: list[str] = [f.strip() for f in _FENCE.findall(text)]
    candidates.append(text.strip())

    for cand in list(candidates):
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            if start == -1:
                continue
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(cand[start : i + 1])
                        break

    for cand in candidates:
        if not cand:
            continue
        for attempt in (cand, _soften(cand)):
            try:
                return json.loads(attempt)
            except Exception:
                continue
    raise ValueError("no parseable JSON in model output")


def _soften(text: str) -> str:
    """Trailing commas and stray control characters are the usual culprits."""
    out = re.sub(r",(\s*[}\]])", r"\1", text)
    out = "".join(ch for ch in out if ch >= " " or ch in "\n\t")
    return out


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class LLMClient:
    """Chat + structured-output wrapper over an OpenAI-compatible or Ollama endpoint."""

    def __init__(self, cfg: ModelConfig, ledger: Ledger, *, verbose: bool = False):
        self.cfg = cfg
        self.ledger = ledger
        self.verbose = verbose
        self._session = requests.Session()

    # -- plumbing ---------------------------------------------------------
    @property
    def available(self) -> bool:
        """False when no real model is configured (deterministic mode)."""
        return self.cfg.provider in {"openai", "ollama"}

    def _post(self, url: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        resp = self._session.post(
            url, headers=headers, json=payload, timeout=self.cfg.request_timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{resp.status_code} from model endpoint: {resp.text[:400]}"
            )
        return resp.json()

    def _call_raw(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        if self.cfg.provider == "ollama":
            data = self._post(
                self.cfg.base_url.rstrip("/") + "/api/chat",
                {
                    "model": self.cfg.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            return (data.get("message") or {}).get("content", "")
        data = self._post(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        return data["choices"][0]["message"].get("content") or ""

    # -- public API -------------------------------------------------------
    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        stage: str = "misc",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """One free-text turn. Returns "" when no model is configured or all retries fail."""
        if not self.available:
            self.ledger.record(CallRecord(stage, 0, 0, 0.0, fell_back=True))
            return ""
        self.ledger.check()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_output_tokens

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.time()
            try:
                text = self._call_raw(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                self.ledger.record(
                    CallRecord(
                        stage,
                        estimate_tokens(system or "") + estimate_tokens(prompt),
                        estimate_tokens(text),
                        time.time() - t0,
                    )
                )
                return text
            except Exception as exc:  # network blip, rate limit, 5xx
                last_err = exc
                self.ledger.record(
                    CallRecord(stage, estimate_tokens(prompt), 0, time.time() - t0)
                )
                if self.verbose:
                    print(f"  [llm] {stage} attempt {attempt + 1} failed: {exc}")
                time.sleep(min(2 ** attempt, 8))
        if self.verbose:
            print(f"  [llm] {stage} giving up: {last_err}")
        return ""

    def structured(
        self,
        prompt: str,
        *,
        system: str | None = None,
        stage: str = "misc",
        validate: Callable[[Any], Any] | None = None,
        fallback: Callable[[], Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Any, str]:
        """Ask for JSON and *guarantee* a usable value.

        Returns ``(value, source)`` where source is ``"model"``, ``"repair"`` or
        ``"fallback"``. ``validate`` may normalise the parsed value or raise to
        reject it, which triggers the repair turn and then the fallback.
        """

        def _use_fallback() -> tuple[Any, str]:
            if fallback is None:
                raise ValueError(f"stage {stage}: no usable JSON and no fallback defined")
            self.ledger.record(CallRecord(stage, 0, 0, 0.0, fell_back=True))
            return fallback(), "fallback"

        if not self.available:
            return _use_fallback()

        sys_prompt = (system or "") + (
            "\n\nReply with JSON only. No prose, no markdown fences, no commentary."
        )
        raw = self.chat(
            prompt,
            system=sys_prompt,
            stage=stage,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parse_error: Exception | None = None
        try:
            value = extract_json(raw)
            return (validate(value) if validate else value), "model"
        except Exception as exc:
            parse_error = exc
            if self.verbose:
                print(f"  [llm] {stage} bad JSON ({exc}); attempting repair")

        # One repair turn: hand the model its own broken output back.
        repair = (
            "Your previous reply was not valid JSON matching the request.\n"
            f"Error: {parse_error}\n\n"
            f"Previous reply:\n{raw[:3000]}\n\n"
            "Return ONLY the corrected JSON value. Nothing else."
        )
        fixed = self.chat(
            repair, system=sys_prompt, stage=stage, temperature=0.0, max_tokens=max_tokens
        )
        if self.ledger.calls:
            self.ledger.calls[-1].repaired = True
        try:
            value = extract_json(fixed)
            return (validate(value) if validate else value), "repair"
        except Exception as exc:
            if self.verbose:
                print(f"  [llm] {stage} repair failed ({exc}); deterministic fallback")
            return _use_fallback()


def first_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        return first_str(value[0], default)
    if value is None:
        return default
    return str(value).strip()


def as_str_list(value: Any, limit: int | None = None) -> list[str]:
    """Coerce whatever the model produced into a clean list of strings."""
    out: list[str] = []
    if isinstance(value, str):
        parts: Iterable[str] = re.split(r"[\n;]+", value)
        out = [p.strip(" -*\t") for p in parts]
    elif isinstance(value, dict):
        out = [f"{k}: {first_str(v)}" for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                out.append(
                    first_str(
                        item.get("topic")
                        or item.get("name")
                        or item.get("text")
                        or item.get("question")
                        or item
                    )
                )
            else:
                out.append(first_str(item))
    cleaned = [o for o in (s.strip() for s in out) if o]
    seen: set[str] = set()
    deduped: list[str] = []
    for o in cleaned:
        key = o.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(o)
    return deduped[:limit] if limit else deduped
