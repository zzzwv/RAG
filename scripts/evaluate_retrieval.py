from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_app.runtime import build_runtime


def hit_at_3(results, relevant_sources: set[str]) -> int:
    return int(any(item.chunk.source in relevant_sources for item in results[:3]))


def main() -> None:
    parser = argparse.ArgumentParser(description="对比向量基线与混合重排 Recall@3")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime = build_runtime()
    vector_hits = 0
    hybrid_hits = 0
    for row in rows:
        relevant = set(row["relevant_sources"])
        vector_hits += hit_at_3(runtime.vector_store.search(row["question"], top_k=3), relevant)
        candidates = runtime.retriever.search(row["question"])
        hybrid_hits += hit_at_3(runtime.reranker.rerank(row["question"], candidates), relevant)
    count = len(rows) or 1
    baseline = vector_hits / count
    enhanced = hybrid_hits / count
    print(json.dumps({
        "questions": len(rows), "vector_recall_at_3": baseline,
        "hybrid_rerank_recall_at_3": enhanced, "absolute_improvement": enhanced - baseline,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
