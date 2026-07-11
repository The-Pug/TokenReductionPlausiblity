import ast
import operator
import re

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000 or (abs(left) > 1 and abs(right) * len(str(abs(left))) > 4000):
                raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def format_number(v: float) -> str:
    return str(int(v)) if float(v) == int(v) else f"{v:g}"


_EXPR_RE = re.compile(
    r"(?:what is|calculate|compute|evaluate)\s+([\d\s()+\-*/×÷^.,%]+?)\s*[?.]?\s*$",
    re.I)


def solve_arithmetic(prompt: str) -> str | None:
    m = _EXPR_RE.search(prompt.strip())
    if not m:
        return None
    expr = (m.group(1).replace("×", "*").replace("÷", "/")
            .replace("^", "**").replace(",", ""))
    if expr.count("%"):
        return None
    if not re.fullmatch(r"[\d\s()+\-*/.**]+", expr) or not re.search(r"\d", expr):
        return None
    try:
        value = _safe_eval(ast.parse(expr, mode="eval"))
        answer = format_number(value)
    except Exception:
        return None
    return f"{expr.strip()} = {answer}"


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text.replace("**", ""))
    if not nums:
        return None
    try:
        return format_number(float(nums[-1].replace(",", "")))
    except ValueError:
        return None


def python_compiles(text: str) -> bool:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    candidates = blocks or [text]
    for c in candidates:
        try:
            ast.parse(c)
            return True
        except SyntaxError:
            continue
    return False


_LEN_SENT = re.compile(r"in (one|1|two|2|three|3) sentences?", re.I)
_LEN_WORDS = re.compile(r"in (?:at most |under |no more than )?(\d+) words", re.I)
_WORDS_TO_N = {"one": 1, "1": 1, "two": 2, "2": 2, "three": 3, "3": 3}


def meets_length_constraint(prompt: str, answer: str) -> bool:
    m = _LEN_SENT.search(prompt)
    if m:
        n = _WORDS_TO_N[m.group(1).lower()]
        sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", answer.strip()) if s.strip()]
        return len(sentences) <= n
    m = _LEN_WORDS.search(prompt)
    if m:
        return len(answer.split()) <= int(m.group(1)) * 1.1
    return True
