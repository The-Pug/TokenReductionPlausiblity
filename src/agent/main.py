import asyncio
import json
import os
import sys
import time

from .cascade import Agent
from .clients import ModelTiers, TokenLedger, fireworks_client, local_model

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
TIME_BUDGET = float(os.environ.get("TIME_BUDGET_SECONDS", 540))
RESERVE = 60
LOCAL_CONCURRENCY = int(os.environ.get("LOCAL_CONCURRENCY", 2))


async def run(tasks: list[dict]) -> list[dict]:
    start = time.monotonic()
    ledger = TokenLedger()
    agent = Agent(local_model(), fireworks_client(), ModelTiers(), ledger)
    local_sem = asyncio.Semaphore(LOCAL_CONCURRENCY)
    results: dict[str, str] = {t["task_id"]: "" for t in tasks}

    async def one(t: dict):
        tid, prompt = t["task_id"], t.get("prompt", "")
        async with local_sem:
            pressed = time.monotonic() - start > TIME_BUDGET - RESERVE
            try:
                r = await agent.solve(tid, prompt, force_remote=pressed)
                results[tid] = r.answer
            except Exception as e:
                print(f"[{tid}] unhandled: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)

    try:
        await asyncio.wait_for(asyncio.gather(*(one(t) for t in tasks)),
                               timeout=TIME_BUDGET)
    except asyncio.TimeoutError:
        print("global time budget exhausted; flushing partial results",
              file=sys.stderr, flush=True)

    print(f"done in {time.monotonic()-start:.0f}s | remote calls={ledger.calls} "
          f"tokens={ledger.total} (prompt={ledger.prompt} completion={ledger.completion})",
          file=sys.stderr, flush=True)
    return [{"task_id": tid, "answer": ans} for tid, ans in results.items()]


def write_results(payload: list[dict]):
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, OUTPUT_PATH)


def main() -> int:
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"cannot read {INPUT_PATH}: {e}", file=sys.stderr)
        write_results([])
        return 1

    payload = asyncio.run(run(tasks))
    write_results(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
