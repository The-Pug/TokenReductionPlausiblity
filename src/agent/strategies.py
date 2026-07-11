import os
import re
from dataclasses import dataclass
from typing import Callable

from . import solvers

Verify = Callable[[str, str], bool]


@dataclass(frozen=True)
class Strategy:
    system: str
    verify: Verify
    samples: int = 1
    agree_on: str | None = None
    local_max_tokens: int = 320
    escalate_default: bool = False
    escalate_if: Callable[[str], bool] | None = None
    tier: str = "large"
    remote_max_tokens: int = 320


def _always(prompt: str, text: str) -> bool:
    return bool(text.strip())


def _verify_sentiment(prompt: str, text: str) -> bool:
    return bool(re.search(r"\b(positive|negative|neutral|mixed)\b", text, re.I))


def _verify_ner(prompt: str, text: str) -> bool:
    return bool(re.search(
        r"\b(person|people|organi[sz]ation|company|location|place|date)s?\b\s*[:\-–]",
        text, re.I))


def _verify_summary(prompt: str, text: str) -> bool:
    return bool(text.strip()) and solvers.meets_length_constraint(prompt, text)


def _verify_code(prompt: str, text: str) -> bool:
    python_asked = "python" in prompt.lower() or "def " in prompt
    if python_asked:
        return solvers.python_compiles(text)
    return "```" in text or bool(text.strip())


def _verify_math(prompt: str, text: str) -> bool:
    return solvers.final_number(text) is not None


_MATH_TRAP = re.compile(
    r"original (price|value|cost)|reverse.{0,15}percent|grows?\s+\d+(?:\.\d+)?%|"
    r"compound|project(?:ion)?\b", re.I)


def _math_is_trappy(prompt: str) -> bool:
    return bool(_MATH_TRAP.search(prompt))


SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)

STRATEGIES: dict[str, Strategy] = {
    "factual": Strategy(
        system=("Answer directly and accurately in English. Give the key facts "
                "in 2-4 sentences; no preamble."),
        verify=_always, samples=1, local_max_tokens=260,
        tier="gemma", remote_max_tokens=400),
    "math": Strategy(
        system=("Solve step by step, briefly. End with a final line: "
                "'Answer: <number>'."),
        verify=_verify_math, samples=2, agree_on="number", local_max_tokens=400,
        escalate_if=_math_is_trappy, tier="large", remote_max_tokens=500),
    "sentiment": Strategy(
        system=("Classify the sentiment as positive, negative, neutral, or mixed. "
                "State the label first, then justify it in one sentence."),
        verify=_verify_sentiment, samples=2, agree_on="label", local_max_tokens=140,
        tier="gemma", remote_max_tokens=300),
    "summarization": Strategy(
        system=("Summarise faithfully and obey any format or length constraint "
                "exactly. Output only the summary."),
        verify=_verify_summary, samples=1, local_max_tokens=220,
        tier="gemma", remote_max_tokens=350),
    "ner": Strategy(
        system=("Extract the named entities and group them under the labels "
                "Person, Organization, Location, Date. One label per line, "
                "entities comma-separated. Only list entities present."),
        verify=_verify_ner, escalate_default=True, local_max_tokens=200,
        tier="gemma", remote_max_tokens=350),
    "code_debugging": Strategy(
        system=("Identify the bug precisely, then provide the corrected code in "
                "a fenced block, followed by a one-sentence explanation."),
        verify=_verify_code, escalate_default=True, local_max_tokens=500,
        tier="code", remote_max_tokens=750),
    "logic": Strategy(
        system=("Reason carefully through the constraints step by step, check "
                "every condition, then state the final answer clearly on the "
                "last line as 'Answer: <answer>'."),
        verify=_always, escalate_default=True, local_max_tokens=400,
        tier="large", remote_max_tokens=600),
    "code_generation": Strategy(
        system=("Write clean, correct, well-structured code exactly to the spec, "
                "in a fenced block. Include a docstring; no usage examples unless "
                "asked."),
        verify=_verify_code, escalate_default=True, local_max_tokens=500,
        tier="code", remote_max_tokens=750),
}


def get_strategy(category: str) -> Strategy:
    s = STRATEGIES[category]
    local_first = {c.strip() for c in os.environ.get("LOCAL_FIRST_CATEGORIES", "").split(",") if c.strip()}
    escalate = {c.strip() for c in os.environ.get("ESCALATE_CATEGORIES", "").split(",") if c.strip()}
    if category in local_first and (s.escalate_default or s.escalate_if):
        s = Strategy(**{**s.__dict__, "escalate_default": False, "escalate_if": None})
    elif category in escalate and not s.escalate_default:
        s = Strategy(**{**s.__dict__, "escalate_default": True})
    return s


def agreement_signal(kind: str | None, text: str) -> str | None:
    if kind == "number":
        return solvers.final_number(text)
    if kind == "label":
        m = SENTIMENT_LABEL.search(text)
        return m.group(1).lower() if m else None
    return text.strip().lower() or None
