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
    api_error: str = ""      # transport/HTTP failure, after retries were spent
    truncated: bool = False  # provider stopped us at the output-token limit


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

    @property
    def api_errors(self) -> int:
        return sum(1 for c in self.calls if c.api_error)

    @property
    def truncations(self) -> int:
        return sum(1 for c in self.calls if c.truncated)

    def error_samples(self, limit: int = 3) -> list[str]:
        seen: list[str] = []
        for c in self.calls:
            if c.api_error and c.api_error not in seen:
                seen.append(c.api_error)
            if len(seen) >= limit:
                break
        return seen

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
                {"calls": 0, "tokens": 0, "seconds": 0.0, "repairs": 0,
                 "fallbacks": 0, "api_errors": 0, "truncations": 0},
            )
            s["calls"] += 1
            s["tokens"] += c.prompt_tokens + c.output_tokens
            s["seconds"] += c.seconds
            s["repairs"] += int(c.repaired)
            s["fallbacks"] += int(c.fell_back)
            s["api_errors"] += int(bool(c.api_error))
            s["truncations"] += int(c.truncated)
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
            "api_errors": self.api_errors,
            "api_error_samples": self.error_samples(),
            "output_truncations": self.truncations,
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



class RateLimiter:
    """Client-side sliding-window token throttle.

    Free model endpoints meter tokens per minute (Groq gives 8,000 TPM on the
    tier this was developed against). Reacting to 429s wastes a round trip and
    a retry each time; it is cheaper and far faster to simply not exceed the
    limit. This tracks the tokens actually spent in the trailing 60 seconds and
    sleeps just long enough before a call that would breach the budget.

    `tokens_per_minute=0` disables it, which is what a local Ollama wants.
    """

    def __init__(self, tokens_per_minute: int, *, safety: float = 0.88,
                 verbose: bool = False):
        self.limit = max(0, int(tokens_per_minute))
        self.safety = safety
        self.verbose = verbose
        self._events: list[tuple[float, int]] = []

    @property
    def budget(self) -> float:
        return self.limit * self.safety

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._events = [(t, n) for t, n in self._events if t > cutoff]

    def acquire(self, estimated: int) -> float:
        """Block until `estimated` tokens fit in the window. Returns seconds slept."""
        if not self.limit:
            return 0.0
        slept = 0.0
        while True:
            now = time.time()
            self._prune(now)
            used = sum(n for _, n in self._events)
            if used + estimated <= self.budget or not self._events:
                return slept
            oldest = self._events[0][0]
            wait = max(0.25, 60.0 - (now - oldest) + 0.2)
            if self.verbose:
                print(
                    f"  [llm] throttling {wait:.1f}s "
                    f"({used} tokens used in the last minute, need {estimated})"
                )
            time.sleep(wait)
            slept += wait

    def record(self, actual: int) -> None:
        if self.limit:
            self._events.append((time.time(), max(0, int(actual))))


class LLMClient:
    """Chat + structured-output wrapper over an OpenAI-compatible or Ollama endpoint."""

    def __init__(self, cfg: ModelConfig, ledger: Ledger, *, verbose: bool = False):
        self.cfg = cfg
        self.ledger = ledger
        self.verbose = verbose
        self._session = requests.Session()
        self.limiter = RateLimiter(cfg.tokens_per_minute, verbose=verbose)

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

    def _call_raw(
        self, messages: list[dict], *, temperature: float, max_tokens: int
    ) -> tuple[str, str, int]:
        """Returns (text, finish_reason, tokens_used). "length" means truncated."""
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
            text = (data.get("message") or {}).get("content", "")
            used = int(data.get("prompt_eval_count") or 0) + int(data.get("eval_count") or 0)
            return text, ("length" if data.get("done_reason") == "length" else "stop"), used

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Open reasoning models (gpt-oss and friends) spend hidden reasoning
        # tokens out of the same output budget. Asking for low effort keeps the
        # budget for the answer we actually parse. Ignored by other providers.
        if "gpt-oss" in self.cfg.model or "qwen3" in self.cfg.model:
            payload["reasoning_effort"] = "low"
        data = self._post(self.cfg.base_url.rstrip("/") + "/chat/completions", payload)
        choice = data["choices"][0]
        used = int((data.get("usage") or {}).get("total_tokens") or 0)
        return (
            (choice["message"].get("content") or ""),
            (choice.get("finish_reason") or "stop"),
            used,
        )

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
        budget_tokens = max_tokens
        for attempt in range(self.cfg.max_retries):
            prompt_estimate = estimate_tokens(system or "") + estimate_tokens(prompt)
            # Reserve the output allowance too: the meter counts what comes back.
            self.limiter.acquire(prompt_estimate + min(budget_tokens, 1200))
            t0 = time.time()
            try:
                text, finish, used = self._call_raw(
                    messages, temperature=temperature, max_tokens=budget_tokens
                )
                truncated = finish == "length"
                self.limiter.record(used or (prompt_estimate + estimate_tokens(text)))
                # Prefer the provider's own accounting over our estimate.
                if used:
                    out_tokens = max(0, used - prompt_estimate)
                else:
                    out_tokens = estimate_tokens(text)
                self.ledger.record(
                    CallRecord(
                        stage,
                        prompt_estimate,
                        out_tokens,
                        time.time() - t0,
                        truncated=truncated,
                    )
                )
                # A reply cut off at the token limit is almost never parseable.
                # Retry once with a bigger budget rather than sending a
                # half-written JSON object down the pipeline.
                if truncated and attempt < self.cfg.max_retries - 1:
                    budget_tokens = min(int(budget_tokens * 1.8), 8000)
                    if self.verbose:
                        print(
                            f"  [llm] {stage} output truncated; retrying with "
                            f"max_tokens={budget_tokens}"
                        )
                    continue
                return text
            except Exception as exc:  # network blip, rate limit, 5xx
                last_err = exc
                self.ledger.record(
                    CallRecord(
                        stage, estimate_tokens(prompt), 0, time.time() - t0,
                        api_error=str(exc)[:200],
                    )
                )
                if self.verbose:
                    print(f"  [llm] {stage} attempt {attempt + 1} failed: {exc}")
                # Rate limits need real backoff, and the provider usually says
                # exactly how long to wait - believe it rather than guessing.
                msg = str(exc)
                wait = min(2 ** attempt, 15)
                if "429" in msg or "rate" in msg.lower():
                    hint = re.search(r"try again in ([\d.]+)s", msg)
                    wait = (float(hint.group(1)) + 1.0) if hint else max(wait, 12)
                    self.limiter.record(int(self.limiter.budget))  # assume full window
                time.sleep(wait)
        if self.verbose:
            print(f"  [llm] {stage} giving up after {self.cfg.max_retries}: {last_err}")
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
