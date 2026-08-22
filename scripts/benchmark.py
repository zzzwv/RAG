from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.chat.memory import WindowMemory
from rag_app.runtime import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 热态性能测试")
    parser.add_argument("question")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    runtime = build_runtime()
    service = runtime.create_qa(WindowMemory(runtime.settings.memory_turns))
    service.answer(args.question)  # warm-up; model download/load is excluded
    runs = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        result = service.answer(args.question)
        runs.append({**result.timings, "wall": time.perf_counter() - started})
    averages = {key: sum(run.get(key, 0.0) for run in runs) / len(runs) for key in runs[0]}
    print(json.dumps({"runs": runs, "average": averages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
