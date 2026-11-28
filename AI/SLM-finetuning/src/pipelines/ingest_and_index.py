import argparse
from pathlib import Path
from tqdm import tqdm

from src.core.config import load_config
from src.core.io_utils import write_jsonl
from src.core.pdf_processor import chunk_pdf
from src.core.vector_store import FaissVectorStore


def run(pdf_dir: Path, group: str) -> None:
    cfg = load_config()
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir.resolve()}")

    chunks = []
    for pdf_path in tqdm(pdf_files, desc=f"Chunking [{group}]"):
        chunks.extend(
            chunk_pdf(
                pdf_path,
                source_group=group,
                chunk_size=cfg.chunk_size_words,
                overlap=cfg.chunk_overlap_words,
                min_page_words=cfg.min_page_words,
                min_tail_words=cfg.min_tail_words,
            )
        )

    chunks_path = cfg.data_dir / "chunks" / f"{group}_chunks.jsonl"
    write_jsonl(chunks_path, (c.to_dict() for c in chunks))
    print(f"Saved {len(chunks)} chunks -> {chunks_path}")

    store = FaissVectorStore(model_name=cfg.embed_model)
    index_path = cfg.data_dir / "chunks" / f"{group}_faiss.index"
    total = store.build_and_save([c.text for c in chunks], index_path)
    print(f"Built FAISS index ({total} vectors on {store.device.upper()}) -> {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk PDFs and build FAISS index.")
    parser.add_argument("--pdf_dir", type=Path, default=Path("data/general_pdfs"))
    parser.add_argument("--group", type=str, default="general")
    args = parser.parse_args()
    run(args.pdf_dir, args.group)
