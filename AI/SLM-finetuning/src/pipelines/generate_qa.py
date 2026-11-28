import argparse
import asyncio
from pathlib import Path
from typing import Any
from tqdm.asyncio import tqdm

from src.core.config import load_config
from src.core.io_utils import append_jsonl, read_jsonl
from src.core.llm_client import AsyncLLMClient
from src.prompts import PERSONAS


async def process_chunk(
    client: AsyncLLMClient, chunk: dict[str, Any], out_path: Path
) -> int:
    user_prompt = (
        f"SOURCE FILE: {chunk['source_file']} (Page {chunk['page']})\n\n"
        f"TECHNICAL EXCERPT:\n{chunk['text']}"
    )
    try:
        items = await client.generate_json(
            system_prompt=PERSONAS.generator,
            user_prompt=user_prompt,
            temperature=0.3,
        )
        if not isinstance(items, list):
            return 0

        records = []
        for idx, qa in enumerate(items, start=1):
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if q and a:
                records.append({
                    "qa_id": f"{chunk['chunk_id']}_q{idx:02d}",
                    "chunk_id": chunk["chunk_id"],
                    "source_group": chunk["source_group"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "category": str(qa.get("category", "general")).strip(),
                    "question": q,
                    "answer": a,
                })

        if records:
            append_jsonl(out_path, records)
        return len(records)
    except Exception as exc:
        print(f"[WARN] Skipped {chunk['chunk_id']}: {exc}")
        return 0


async def run(group: str, limit: int | None) -> None:
    cfg = load_config()
    chunks_path = cfg.data_dir / "chunks" / f"{group}_chunks.jsonl"
    out_path = cfg.data_dir / "raw_qa" / f"{group}_raw_qa.jsonl"

    all_chunks = read_jsonl(chunks_path)
    if not all_chunks:
        raise FileNotFoundError(f"No chunks found at {chunks_path}")

    done_ids = {rec["chunk_id"] for rec in read_jsonl(out_path)}
    pending = [c for c in all_chunks if c["chunk_id"] not in done_ids]
    if limit is not None:
        pending = pending[:limit]

    print(f"Group [{group}] | Total chunks: {len(all_chunks)} | Already done: {len(done_ids)} | Running: {len(pending)}")
    if not pending:
        print("Nothing left to generate.")
        return

    client = AsyncLLMClient(cfg.generator, max_concurrency=cfg.max_concurrency)
    try:
        tasks = [process_chunk(client, chunk, out_path) for chunk in pending]
        counts = await tqdm.gather(*tasks, desc=f"Generating QA ({cfg.generator.model})")
        print(f"Done! Generated {sum(counts)} QA pairs -> {out_path}")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs using LLM #1.")
    parser.add_argument("--group", type=str, default="general")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process (use 5 for smoke test)")
    args = parser.parse_args()
    asyncio.run(run(args.group, args.limit))
