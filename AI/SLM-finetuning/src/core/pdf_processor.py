import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import pymupdf

_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]+)-\s+([A-Za-z]+)")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_group: str
    source_file: str
    page: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_page_text(raw_text: str) -> str:
    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip() and not (line.strip().isdigit() and len(line.strip()) <= 4)
    ]
    joined = " ".join(lines)
    return _MULTI_SPACE_RE.sub(" ", _HYPHEN_BREAK_RE.sub(r"\1\2", joined)).strip()


def iter_clean_pages(pdf_path: Path, min_words: int = 35) -> Iterator[tuple[int, str]]:
    with pymupdf.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            cleaned = clean_page_text(page.get_text("text"))
            if len(cleaned.split()) >= min_words:
                yield page_idx, cleaned


def chunk_pdf(
    pdf_path: Path,
    source_group: str,
    chunk_size: int = 450,
    overlap: int = 75,
    min_page_words: int = 35,
    min_tail_words: int = 45,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    chunk_idx = 0

    for page_num, text in iter_clean_pages(pdf_path, min_words=min_page_words):
        words = text.split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if len(window) < min_tail_words and chunks:
                chunks[-1].text = f"{chunks[-1].text} {' '.join(window)}"
                break

            chunk_idx += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{pdf_path.stem}_c{chunk_idx:04d}",
                    source_group=source_group,
                    source_file=pdf_path.name,
                    page=page_num,
                    text=" ".join(window),
                )
            )
            if start + chunk_size >= len(words):
                break

    return chunks
