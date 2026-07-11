import re

CATEGORIES = (
    "code_debugging", "code_generation", "sentiment", "ner",
    "summarization", "logic", "math", "factual",
)

_CODE_HINT = re.compile(
    r"```|\bdef |\bclass |\bfunction\b|\breturn\b|;\s*$|\bimport |#include|console\.log",
    re.M)
_BUG_HINT = re.compile(
    r"\b(bug|debug|fix|error|incorrect|wrong|fails?|broken|doesn'?t work|not work)",
    re.I)
_GEN_HINT = re.compile(
    r"\b(write|implement|create|generate)\b.{0,60}\b(function|method|class|program|script|code)\b",
    re.I)
_SENT_HINT = re.compile(
    r"\bsentiment\b|classify.{0,40}(review|tweet|comment|text|tone)|positive.{0,20}negative",
    re.I)
_NER_HINT = re.compile(
    r"named entit|\bNER\b|(extract|identify|list|find).{0,50}\b(entities|people|persons?|organi[sz]ations?|locations?|dates?)\b.{0,60}\b(from|in)\b",
    re.I)
_SUM_HINT = re.compile(
    r"\bsummari[sz]e|\bsummary\b|\btl;?dr\b|condense|in (one|1|two|2|three|3) sentences?|in \d+ words",
    re.I)
_LOGIC_HINT = re.compile(
    r"\b(puzzle|riddle|constraints?|deduce|deduction|who (sits|owns|lives|came)|seating|"
    r"knights?|knaves|liars?|truth-?teller|always (lies?|tells?)|must be true|arrange|ordering|"
    r"finished (before|after)|came (first|last)|left of|right of|next to|"
    r"older than|younger than)\b", re.I)
_MATH_HINT = re.compile(
    r"\b(calculate|compute|how (many|much)|total|sum|difference|product|remainder|"
    r"percent|percentage|average|speed|price|cost|profit|interest|probability|"
    r"revenue|project(?:ion)?|grows?|growth|increases?|decreases?)\b", re.I)
_ARITH = re.compile(r"\d[\d,.]*\s*[-+*/×÷^%]\s*\d|\d+(?:\.\d+)?\s*%")
_FACT_HINT = re.compile(r"\b(what is|what are|explain|define|describe|how does|why (is|do|does))\b", re.I)


def classify(prompt: str) -> str:
    has_code = bool(_CODE_HINT.search(prompt))
    if has_code and _BUG_HINT.search(prompt):
        return "code_debugging"
    if _GEN_HINT.search(prompt) or (has_code and not _BUG_HINT.search(prompt)):
        return "code_generation"
    if _SENT_HINT.search(prompt):
        return "sentiment"
    if _NER_HINT.search(prompt):
        return "ner"
    if _SUM_HINT.search(prompt):
        return "summarization"
    if _LOGIC_HINT.search(prompt) and not _ARITH.search(prompt):
        return "logic"
    if _ARITH.search(prompt) or (_MATH_HINT.search(prompt) and re.search(r"\d", prompt)):
        return "math"
    if _FACT_HINT.search(prompt):
        return "factual"
    return "factual"
