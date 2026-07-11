import sys
import time
from dataclasses import dataclass, field

from . import solvers
from .classify import classify
from .clients import Model, ModelTiers, TokenLedger
from .strategies import agreement_signal, get_strategy


@dataclass
class TaskResult:
    task_id: str
    answer: str
    category: str
    route: str
    remote_tokens: int = 0
    elapsed: float = 0.0
    detail: str = ""
    local_texts: list = field(default_factory=list)


class Agent:
    def __init__(self, local: Model, remote: Model | None, tiers: ModelTiers,
                 ledger: TokenLedger, log=True):
        self.local = local
        self.remote = remote
        self.tiers = tiers
        self.ledger = ledger
        self.log = log

    def _log(self, msg: str):
        if self.log:
            print(msg, file=sys.stderr, flush=True)

    async def _local_attempt(self, prompt: str, strat) -> tuple[str, bool, str]:
        texts, signals = [], []
        for i in range(strat.samples):
            try:
                c = await self.local.complete(
                    prompt, system=strat.system,
                    temperature=0.0 if i == 0 else 0.7,
                    max_tokens=strat.local_max_tokens)
            except Exception as e:
                return ("", False, f"local_error:{type(e).__name__}")
            texts.append(c.text)
            signals.append(agreement_signal(strat.agree_on, c.text))

        best = texts[0]
        if not strat.verify(prompt, best):
            return (best, False, "verify_failed")
        if strat.samples > 1:
            if None in signals or len(set(signals)) != 1:
                return (best, False, f"disagree:{signals}")
        return (best, True, "ok")

    async def _remote_attempt(self, prompt: str, strat) -> tuple[str, int, str]:
        model = self.remote.with_model(self.tiers.pick(strat.tier)) \
            if self.tiers.ids else self.remote
        c = await model.complete(prompt, system=strat.system,
                                 temperature=0.0,
                                 max_tokens=strat.remote_max_tokens)
        await self.ledger.add(c)
        return (c.text, c.total_tokens, model.model)

    async def solve(self, task_id: str, prompt: str,
                    force_remote: bool = False) -> TaskResult:
        t0 = time.monotonic()
        category = classify(prompt)
        strat = get_strategy(category)

        def done(answer, route, tokens=0, detail="", local_texts=None):
            r = TaskResult(task_id=task_id, answer=answer.strip(), category=category,
                           route=route, remote_tokens=tokens,
                           elapsed=time.monotonic() - t0, detail=detail,
                           local_texts=local_texts or [])
            self._log(f"[{task_id}] {category:<15} {route:<8} "
                      f"tok={tokens:<5} {r.elapsed:5.1f}s {detail}")
            return r

        if category == "math" and not force_remote:
            solved = solvers.solve_arithmetic(prompt)
            if solved is not None:
                return done(solved, "solver")

        skip_local = strat.escalate_default or (strat.escalate_if and strat.escalate_if(prompt))
        local_text = ""
        if not force_remote and not skip_local:
            local_text, ok, detail = await self._local_attempt(prompt, strat)
            if ok:
                return done(local_text, "local", detail=detail)

        if self.remote is not None:
            try:
                text, tokens, model_id = await self._remote_attempt(prompt, strat)
                if text.strip():
                    return done(text, "remote", tokens, detail=model_id)
            except Exception as e:
                self._log(f"[{task_id}] remote failed: {type(e).__name__}: {e}")

        if not local_text:
            try:
                local_text, _, _ = await self._local_attempt(prompt, strat)
            except Exception:
                local_text = ""
        if local_text.strip():
            return done(local_text, "fallback", detail="unverified_local")

        try:
            c = await self.local.complete(
                prompt, system="Answer concisely and directly.",
                temperature=0.0, max_tokens=120)
            if c.text.strip():
                return done(c.text, "fallback", detail="short_retry")
        except Exception:
            pass
        return done("", "empty", detail="all_stages_failed")
