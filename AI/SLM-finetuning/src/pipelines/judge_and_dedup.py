import argparse
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm

from src.core.config import load_config
from src.core.io_utils import read_jsonl, write_jsonl
from src.core.llm_client import AsyncLLMClient
from src.core.vector_store import FaissVectorStore
from src.prompts import PERSONAS

_DEFAULT_JUDGE_PROMPT = (
    "You are a strict technical QA reviewer. Given a SOURCE EXCERPT, a QUESTION, and an ANSWER, "
    "verify that the answer is factually accurate, complete, and strictly grounded in the excerpt. "
    'Return ONLY valid JSON: {"pass": true/false, "score": 1-5, "reason": "short explanation"}'
)

_DEDUP_PROMPT = (
    "You are a dataset deduplication reviewer. Given Question A and Question B, determine if they "
    "ask essentially the same thing and would be redundant in a fine-tuning dataset. "
    'Return ONLY valid JSON: {"duplicate": true/false}'
)


async def judge_single(
    client: AsyncLLMClient,
    qa: dict[str, Any],
    chunk_text: str,
    judge_prompt: str,
) -> dict[str, Any] | None:
    user_prompt = (
        f"SOURCE EXCERPT:\n{chunk_text}\n\n"
        f"QUESTION:\n{qa['question']}\n\n"
        f"ANSWER:\n{qa['answer']}"
    )
    try:
        res = await client.generate_json(
            system_prompt=judge_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        if not isinstance(res, dict):
            return None

        passed = res.get("pass", res.get("keep"))
        score = res.get("score", 5 if passed else 0)
        is_ok = bool(passed) if passed is not None else (float(score) >= 4.0)
        if not is_ok:
            return None

        return {**qa, "judge_score": score, "judge_reason": str(res.get("reason", "")).strip()}
    except Exception as exc:
        print(f"[WARN] Judge skipped {qa['qa_id']}: {exc}")
        return None


async def deduplicate(
    records: list[dict[str, Any]],
    store: FaissVectorStore,
    client: AsyncLLMClient,
    auto_drop: float,
    judge_min: float,
) -> list[dict[str, Any]]:
    if len(records) <= 1:
        return records

    q_vecs = store.encode([r["question"] for r in records], show_progress=False)
    kept: list[dict[str, Any]] = []
    kept_vecs: list[np.ndarray] = []

    for rec, vec in zip(records, q_vecs):
        if not kept_vecs:
            kept.append(rec)
            kept_vecs.append(vec)
            continue

        sims = np.dot(np.vstack(kept_vecs), vec)
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])

        if max_sim >= auto_drop:
            continue

        if max_sim >= judge_min:
            try:
                res = await client.generate_json(
                    system_prompt=_DEDUP_PROMPT,
                    user_prompt=f"Question A: {kept[max_idx]['question']}\nQuestion B: {rec['question']}",
                    temperature=0.0,
                )
                if isinstance(res, dict) and res.get("duplicate"):
                    continue
            except Exception:
                pass

        kept.append(rec)
        kept_vecs.append(vec)

    return kept


async def run(group: str) -> None:
    cfg = load_config()
    raw_path = cfg.data_dir / "raw_qa" / f"{group}_raw_qa.jsonl"
    chunks_path = cfg.data_dir / "chunks" / f"{group}_chunks.jsonl"
    index_path = cfg.data_dir / "chunks" / f"{group}_faiss.index"
    out_path = cfg.data_dir / "curated_qa" / f"{group}_curated_qa.jsonl"

    raw_qa = read_jsonl(raw_path)
    if not raw_qa:
        raise FileNotFoundError(f"No raw QA found at {raw_path}")

    chunk_map = {c["chunk_id"]: c["text"] for c in read_jsonl(chunks_path)}

    # 1. FAISS semantic grounding check
    store = FaissVectorStore(model_name=cfg.embed_model)
    store.load(index_path)
    qa_texts = [f"{r['question']} {r['answer']}" for r in raw_qa]
    scores, _ = store.search(qa_texts, top_k=1)

    grounded = []
    for rec, sim in zip(raw_qa, scores[:, 0]):
        if float(sim) >= cfg.grounding_min_sim and rec["chunk_id"] in chunk_map:
            grounded.append({**rec, "grounding_sim": round(float(sim), 4)})

    print(f"[{group}] Raw: {len(raw_qa)} -> Grounded (>={cfg.grounding_min_sim}): {len(grounded)}")

    # 2. LLM #2 Judge + 3. Semantic Deduplication
    judge_prompt = getattr(PERSONAS, "judge", _DEFAULT_JUDGE_PROMPT)
    client = AsyncLLMClient(cfg.judge, max_concurrency=cfg.max_concurrency)
    try:
        tasks = [
            judge_single(client, rec, chunk_map[rec["chunk_id"]], judge_prompt)
            for rec in grounded
        ]
        results = await tqdm.gather(*tasks, desc=f"Judging QA ({cfg.judge.model})")
        judged = [r for r in results if r is not None]
        print(f"[{group}] Judge passed: {len(judged)}/{len(grounded)}")

        curated = await deduplicate(
            judged,
            store=store,
            client=client,
            auto_drop=cfg.dedup_auto_drop,
            judge_min=cfg.dedup_judge_min,
        )
        write_jsonl(out_path, curated)
        print(f"[{group}] Final deduplicated: {len(curated)} -> {out_path}")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter, judge, and deduplicate synthetic QA pairs.")
    parser.add_argument("--group", type=str, default="general")
    args = parser.parse_args()
    asyncio.run(run(args.group))
