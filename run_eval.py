import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

from src.agent import solvers  # noqa: E402
from src.agent.cascade import Agent  # noqa: E402
from src.agent.classify import classify  # noqa: E402
from src.agent.clients import (ModelTiers, TokenLedger, fireworks_client,  # noqa: E402
                               local_model)

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(answer: str) -> str:
    blocks = [b for b in CODE_BLOCK.findall(answer) if "def " in b]
    if blocks:
        return blocks[-1]
    return answer if "def " in answer else ""


def run_code_test(answer: str, test: str) -> bool:
    code = extract_code(answer)
    if not code:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test + "\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=5)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def keyword_match(answer: str, groups) -> bool:
    low = answer.lower()
    return all(any(alt.lower() in low for alt in g.split("|")) for g in groups)


def grade(task: dict, answer: str) -> bool:
    if not answer.strip():
        return False
    cat, ref = task["category"], task.get("reference")
    if cat == "math":
        return solvers.final_number(answer) == ref
    if cat == "sentiment":
        m = re.search(r"\b(positive|negative|neutral|mixed)\b", answer, re.I)
        return bool(m) and m.group(1).lower() == ref
    if cat in ("code_generation", "code_debugging") and task.get("test"):
        return run_code_test(answer, task["test"])
    ok = keyword_match(answer, ref)
    if cat == "summarization":
        ok = ok and solvers.meets_length_constraint(task["prompt"], answer)
    return ok


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/dev_tasks.jsonl")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--local-only", action="store_true",
                    help="stub remote with the local model (no API keys needed)")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["category"] in wanted]
    if args.limit:
        tasks = tasks[:args.limit]

    local = local_model()
    remote = local if args.local_only else fireworks_client()
    ledger = TokenLedger()
    agent = Agent(local, remote, ModelTiers([] if args.local_only else None), ledger)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(t):
        async with sem:
            return await agent.solve(t["task_id"], t["prompt"])

    results = await asyncio.gather(*(one(t) for t in tasks))

    stats = defaultdict(lambda: {"n": 0, "ok": 0, "remote": 0, "tokens": 0})
    misroutes = []
    for t, r in zip(tasks, results):
        ok = grade(t, r.answer)
        s = stats[t["category"]]
        s["n"] += 1
        s["ok"] += ok
        s["remote"] += r.route in ("remote",)
        s["tokens"] += r.remote_tokens
        if classify(t["prompt"]) != t["category"]:
            misroutes.append((t["task_id"], t["category"], classify(t["prompt"])))
        flag = "OK " if ok else "BAD"
        print(f"  {flag} [{t['task_id']:>4}] {t['category']:<15} via {r.route:<8} "
              f"tok={r.remote_tokens}")

    print(f"\n{'category':<16}{'acc':>8}{'escalated':>11}{'tokens':>8}")
    tot_n = tot_ok = tot_rem = tot_tok = 0
    for cat, s in sorted(stats.items()):
        print(f"{cat:<16}{s['ok']}/{s['n']:>4}{s['remote']:>9}{s['tokens']:>10}")
        tot_n += s["n"]; tot_ok += s["ok"]; tot_rem += s["remote"]; tot_tok += s["tokens"]
    print("-" * 43)
    print(f"{'TOTAL':<16}{tot_ok}/{tot_n:>4} = {tot_ok/tot_n:.0%}"
          f"{tot_rem:>6}{tot_tok:>10}")
    print(f"remote calls={ledger.calls}  remote tokens={ledger.total}"
          + ("  (LOCAL-ONLY STUB — token counts not real)" if args.local_only else ""))
    if misroutes:
        print(f"misclassified ({len(misroutes)}): {misroutes}")


if __name__ == "__main__":
    asyncio.run(main())
