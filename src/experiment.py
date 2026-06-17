import argparse
import logging
import statistics
import time
from typing import Dict, List

import pandas as pd
import tiktoken
from pinecone import Pinecone

from consts import EMBEDDING_DIMENSIONS, PINECONE_API_KEY, PINECONE_INDEX
from ingest import DATA_CSV, build_chunk_records, ensure_index, upsert_records
from rag import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

CANDIDATE_CHUNK_SIZES: List[int] = [256, 512, 1024]
OVERLAP_RATIO: float = 0.2
EVAL_TOP_KS: List[int] = [1, 5, 10]

EXAMPLE_QUESTIONS: List[str] = [
    "Find an article that reframes marketing as a conversation with readers, "
    "aimed at writers who find self-promotion uncomfortable.",
    "List exactly 3 articles about education.",
    "Find an article that argues past pandemics (such as the bubonic plague) "
    "can spur innovation and recovery.",
    "I want practical, beginner-friendly advice on building habits that actually stick.",
]


def _held_out_query(text: str, max_tokens: int = 64) -> str:
    # grab a slice from the middle to use as the probe
    tokens = _ENCODING.encode(text)

    if len(tokens) <= max_tokens:
        return text

    start = max(0, len(tokens) // 2 - max_tokens // 2)

    return _ENCODING.decode(tokens[start:start + max_tokens])


def _known_item_eval(eval_articles: pd.DataFrame, namespace: str) -> Dict[str, float]:
    max_k = max(EVAL_TOP_KS)
    hits = {k: 0 for k in EVAL_TOP_KS}
    reciprocal_ranks: List[float] = []

    for article_id, row in eval_articles.iterrows():
        text = str(row["text"]) if pd.notna(row["text"]) else ""

        if not text.strip():
            continue

        probe = _held_out_query(text)
        contexts = retrieve(probe, top_k=max_k, namespace=namespace)
        ranked_ids = [ctx["article_id"] for ctx in contexts]
        target = str(article_id)

        rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in EVAL_TOP_KS:
            if rank and rank <= k:
                hits[k] += 1

    n = len(reciprocal_ranks) or 1
    result = {f"recall@{k}": hits[k] / n for k in EVAL_TOP_KS}
    result["mrr"] = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return result


def _show_example_questions(namespace: str) -> None:
    for question in EXAMPLE_QUESTIONS:
        contexts = retrieve(question, top_k=5, namespace=namespace)
        logger.info("Q: %s", question)
        seen_titles: List[str] = []

        for ctx in contexts:
            title = str(ctx["title"])

            if title not in seen_titles:
                seen_titles.append(title)
                logger.info("    [%.4f] %s", ctx["score"], title)

            if len(seen_titles) >= 3:
                break


def run_experiment(articles_number: int, sample_size: int) -> Dict[int, Dict[str, float]]:
    df = pd.read_csv(DATA_CSV).head(articles_number)
    eval_articles = df.sample(n=min(sample_size, len(df)), random_state=42)

    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pinecone, PINECONE_INDEX, EMBEDDING_DIMENSIONS)
    index = pinecone.Index(PINECONE_INDEX)

    results: Dict[int, Dict[str, float]] = {}
    namespaces: List[str] = []

    try:
        for chunk_size in CANDIDATE_CHUNK_SIZES:
            namespace = f"exp_cs{chunk_size}"
            namespaces.append(namespace)
            overlap_tokens = int(chunk_size * OVERLAP_RATIO)

            logger.info("=== Embedding config chunk_size=%d into '%s' ===", chunk_size, namespace)
            records = build_chunk_records(df, chunk_size, overlap_tokens)
            upsert_records(index, records, namespace=namespace)

            time.sleep(10)  # let the upsert settle (eventual consistency)

            logger.info("--- Known-item retrieval eval (chunk_size=%d) ---", chunk_size)
            results[chunk_size] = _known_item_eval(eval_articles, namespace)

            logger.info("--- Example questions (chunk_size=%d) ---", chunk_size)
            _show_example_questions(namespace)
    finally:
        for namespace in namespaces:
            logger.info("Cleaning up namespace '%s'...", namespace)
            try:
                index.delete(delete_all=True, namespace=namespace)
            except Exception as exc:  # best-effort cleanup
                logger.warning("Could not delete namespace '%s': %s", namespace, exc)

    return results


def _print_summary(results: Dict[int, Dict[str, float]]) -> None:
    header = f"{'chunk_size':>11} | " + " | ".join(
        [f"recall@{k}" for k in EVAL_TOP_KS] + ["mrr"]
    )
    logger.info("SUMMARY")
    logger.info(header)
    logger.info("-" * len(header))

    for chunk_size, metrics in results.items():
        row = f"{chunk_size:>11} | " + " | ".join(
            [f"{metrics[f'recall@{k}']:.3f}   " for k in EVAL_TOP_KS]
            + [f"{metrics['mrr']:.3f}"]
        )
        logger.info(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG chunk-size settings.")
    parser.add_argument("--articles", type=int, default=200)
    parser.add_argument("--sample", type=int, default=25)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_experiment(args.articles, args.sample)
    _print_summary(results)


if __name__ == "__main__":
    main()
