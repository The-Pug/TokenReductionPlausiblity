import asyncio
import os
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", 25))


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Model:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = REQUEST_TIMEOUT):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                                  timeout=timeout, max_retries=1)
        self.model = model

    def with_model(self, model: str) -> "Model":
        m = object.__new__(Model)
        m.client = self.client
        m.model = model
        return m

    async def complete(self, prompt: str, system: str | None = None,
                       temperature: float = 0.0, max_tokens: int = 512) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens)
        u = r.usage
        return Completion(
            text=r.choices[0].message.content or "",
            prompt_tokens=u.prompt_tokens if u else 0,
            completion_tokens=u.completion_tokens if u else 0,
        )


def local_model() -> Model:
    return Model(
        os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1"),
        os.environ.get("LOCAL_API_KEY", "ollama"),
        os.environ.get("LOCAL_MODEL", "gemma3:1b"),
        timeout=float(os.environ.get("LOCAL_REQUEST_TIMEOUT", REQUEST_TIMEOUT)),
    )


def fireworks_client() -> Model | None:
    base = os.environ.get("FIREWORKS_BASE_URL")
    key = os.environ.get("FIREWORKS_API_KEY")
    if base and key:
        models = allowed_models()
        return Model(base, key, models[0] if models else "")
    if os.environ.get("ALLOW_DEV_REMOTE") and os.environ.get("REMOTE_BASE_URL"):
        return Model(os.environ["REMOTE_BASE_URL"],
                     os.environ.get("REMOTE_API_KEY", ""),
                     os.environ.get("REMOTE_MODEL", ""))
    return None


def allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    ids = [m.strip() for m in raw.split(",") if m.strip()]
    if not ids and os.environ.get("REMOTE_MODEL"):
        ids = [os.environ["REMOTE_MODEL"]]
    return ids


_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])")


def _params(model_id: str) -> float:
    m = _PARAMS_RE.search(model_id)
    return float(m.group(1)) if m else 100.0


class ModelTiers:
    def __init__(self, ids: list[str] | None = None):
        ids = ids if ids is not None else allowed_models()
        self.ids = ids
        by_size = sorted(ids, key=lambda i: (_params(i), "gemma" not in i.lower()))
        self.small = os.environ.get("MODEL_SMALL") or (by_size[0] if by_size else "")
        self.large = os.environ.get("MODEL_LARGE") or (by_size[-1] if by_size else "")
        coder = [i for i in ids if re.search(r"coder|codestral|code", i, re.I)]
        self.code = os.environ.get("MODEL_CODE") or (coder[0] if coder else self.large)
        gemma = [i for i in by_size if "gemma" in i.lower()]
        self.gemma = gemma[-1] if gemma else self.small

    def pick(self, tier: str) -> str:
        return {"small": self.small, "large": self.large,
                "code": self.code, "gemma": self.gemma}.get(tier, self.large)


class TokenLedger:
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self._lock = asyncio.Lock()

    async def add(self, c: Completion):
        async with self._lock:
            self.prompt += c.prompt_tokens
            self.completion += c.completion_tokens
            self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt + self.completion
