import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class PipelineConfig:
    generator: LLMConfig
    judge: LLMConfig
    embed_model: str = "BAAI/bge-base-en-v1.5"
    chunk_size_words: int = 450
    chunk_overlap_words: int = 75
    min_page_words: int = 35
    min_tail_words: int = 45
    max_concurrency: int = 3
    grounding_min_sim: float = 0.65
    dedup_auto_drop: float = 0.88
    dedup_judge_min: float = 0.82
    data_dir: Path = Path("data")


def load_config() -> PipelineConfig:
    gen_key = os.getenv("GENERATOR_API_KEY", "").strip()
    judge_key = os.getenv("JUDGE_API_KEY", "").strip()
    if not gen_key or not judge_key:
        raise ValueError("Missing GENERATOR_API_KEY or JUDGE_API_KEY in .env")

    return PipelineConfig(
        generator=LLMConfig(
            api_key=gen_key,
            base_url=os.getenv("GENERATOR_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("GENERATOR_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        ),
        judge=LLMConfig(
            api_key=judge_key,
            base_url=os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("JUDGE_MODEL", "baidu/ernie-5.5-flash-preview:free"),
        ),
    )
