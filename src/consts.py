import os

from dotenv import load_dotenv

load_dotenv()


class MissingConfigError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise MissingConfigError(f"missing env var: {name}")

    return value


# secrets come from .env (local) / Vercel env (prod)
LLMOD_AI_API_KEY: str = _require_env("LLMOD_AI_API_KEY")
LLMOD_AI_URL: str = os.environ.get("LLMOD_AI_URL", "https://api.llmod.ai/v1")

PINECONE_API_KEY: str = _require_env("PINECONE_API_KEY")
PINECONE_INDEX: str = os.environ.get("PINECONE_INDEX", "medium-rag-index")

# gateway model ids
EMBEDDING_MODEL: str = "NBUECSE-text-embedding-3-small"
EMBEDDING_DIMENSIONS: int = 1536
GPT_5_MODEL: str = "NBUECSE-gpt-5-mini"

# RAG params
CHUNK_SIZE: int = 512
OVERLAP_RATIO: float = 0.2
OVERLAP_TOKENS: int = int(CHUNK_SIZE * OVERLAP_RATIO)
TOP_K: int = 12

BATCH_SIZE: int = 50
