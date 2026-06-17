import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import tiktoken
from pinecone import Pinecone, ServerlessSpec

from consts import (
    BATCH_SIZE,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONS,
    OVERLAP_TOKENS,
    PINECONE_API_KEY,
    PINECONE_INDEX,
)
from use_models import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_PRICE_PER_MILLION_TOKENS: float = 0.02

_ENCODING = tiktoken.get_encoding("cl100k_base")
DATA_CSV = Path(__file__).parent / "data" / "medium-english-50mb.csv"


class IngestionError(RuntimeError):
    pass


def chunk_text(text: str, chunk_size: int, overlap_tokens: int) -> List[str]:
    if overlap_tokens >= chunk_size:
        raise IngestionError("overlap must be smaller than chunk_size")

    tokens = _ENCODING.encode(text)
    stride = chunk_size - overlap_tokens

    chunks: List[str] = []

    for start in range(0, len(tokens), stride):
        chunks.append(_ENCODING.decode(tokens[start:start + chunk_size]))

    return chunks


def build_chunk_records(
        df_sample: pd.DataFrame,
        chunk_size: int,
        overlap_tokens: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    for index, row in df_sample.iterrows():
        text = str(row["text"]) if pd.notna(row["text"]) else ""

        if not text.strip():
            continue

        title = str(row["title"]) if pd.notna(row.get("title")) else "Unknown Title"

        for chunk_idx, chunk in enumerate(chunk_text(text, chunk_size, overlap_tokens)):
            metadata = {
                "article_id": str(index),
                "title": title,
                "authors": str(row.get("authors", "")),
                "url": str(row.get("url", "")),
                "tags": str(row.get("tags", "")),
                "timestamp": str(row.get("timestamp", "")),
                "text": chunk,
            }
            records.append(
                {
                    "id": f"article_{index}_chunk_{chunk_idx}",
                    "text_for_embedding": chunk,
                    "metadata": metadata,
                }
            )

    return records


def estimate_cost(records: List[Dict[str, object]]) -> Dict[str, float]:
    total_tokens = sum(
        len(_ENCODING.encode(str(record["text_for_embedding"]))) for record in records
    )
    usd = total_tokens / 1_000_000 * EMBED_PRICE_PER_MILLION_TOKENS

    return {"chunks": float(len(records)), "tokens": float(total_tokens), "usd": usd}


def ensure_index(pinecone: Pinecone, index_name: str, dimension: int) -> None:
    existing = {idx["name"] for idx in pinecone.list_indexes()}

    if index_name in existing:
        logger.info("Pinecone index '%s' already exists.", index_name)

        return

    logger.info("Creating Pinecone index '%s' (dim=%d, cosine)...", index_name, dimension)
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # wait for the new index to come up before upserting
    for _ in range(30):
        status = getattr(pinecone.describe_index(index_name), "status", None)
        ready = getattr(status, "ready", None)

        if ready is None and isinstance(status, dict):
            ready = status.get("ready")

        if ready:
            logger.info("Index '%s' is ready.", index_name)

            return

        time.sleep(2)

    logger.warning("Index '%s' not ready after wait; continuing.", index_name)


def upsert_records(
        index,
        records: List[Dict[str, object]],
        namespace: str,
        batch_size: int = BATCH_SIZE,
) -> int:
    upserted = 0

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [str(record["text_for_embedding"]) for record in batch]

        vectors = get_embeddings(texts)

        index.upsert(
            vectors=[
                {
                    "id": record["id"],
                    "values": vector,
                    "metadata": record["metadata"],
                }
                for record, vector in zip(batch, vectors)
            ],
            namespace=namespace,
        )
        upserted += len(batch)
        logger.info("Upserted %d / %d chunks...", upserted, len(records))

    return upserted


def process_and_upload_articles(
        csv_path: Path,
        articles_number: int = 20,
        namespace: str = "",
        chunk_size: int = CHUNK_SIZE,
        overlap_tokens: int = OVERLAP_TOKENS,
        estimate_only: bool = False,
) -> Dict[str, float]:
    logger.info("Loading %d articles from %s", articles_number, csv_path)
    df = pd.read_csv(csv_path)
    df_sample = df.head(articles_number)

    records = build_chunk_records(df_sample, chunk_size, overlap_tokens)
    estimate = estimate_cost(records)
    logger.info(
        "Prepared %d chunks (~%d tokens, est. >= $%.4f to embed).",
        int(estimate["chunks"]),
        int(estimate["tokens"]),
        estimate["usd"],
    )

    if estimate_only:
        logger.info("Estimate-only run; nothing embedded.")

        return estimate

    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pinecone, PINECONE_INDEX, EMBEDDING_DIMENSIONS)
    index = pinecone.Index(PINECONE_INDEX)

    upserted = upsert_records(index, records, namespace=namespace)
    logger.info(
        "Done. Upserted %d vectors into index '%s' namespace '%s'.",
        upserted,
        PINECONE_INDEX,
        namespace or "(default)",
    )

    return estimate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Medium articles into Pinecone.")
    parser.add_argument("--articles", type=int, default=20)
    parser.add_argument("--namespace", type=str, default="")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap-tokens", type=int, default=OVERLAP_TOKENS)
    parser.add_argument("--estimate", action="store_true", help="print cost only, no embedding")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_and_upload_articles(
        DATA_CSV,
        articles_number=args.articles,
        namespace=args.namespace,
        chunk_size=args.chunk_size,
        overlap_tokens=args.overlap_tokens,
        estimate_only=args.estimate,
    )


if __name__ == "__main__":
    main()
