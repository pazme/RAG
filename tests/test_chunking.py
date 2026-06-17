import pandas as pd
import pytest

from ingest import (
    IngestionError,
    build_chunk_records,
    chunk_text,
    estimate_cost,
)


def test_chunk_text_short_text_returns_single_chunk() -> None:
    chunks = chunk_text("hello world", chunk_size=512, overlap_tokens=102)
    assert chunks == ["hello world"]


def test_chunk_text_empty_text_returns_no_chunks() -> None:
    assert chunk_text("", chunk_size=512, overlap_tokens=102) == []


def test_chunk_text_overlaps_consecutive_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(60))
    chunks = chunk_text(text, chunk_size=10, overlap_tokens=4)

    assert len(chunks) > 1
    # consecutive chunks share content
    assert chunks[0].split()[-1] in chunks[1]


def test_chunk_text_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(IngestionError):
        chunk_text("anything", chunk_size=100, overlap_tokens=100)


def test_build_chunk_records_skips_blank_text_and_sets_metadata() -> None:
    df = pd.DataFrame(
        {
            "title": ["Real Title", "Blank One"],
            "text": ["Some real article body about testing.", "   "],
            "authors": ["Alice", "Bob"],
            "url": ["http://a", "http://b"],
            "tags": ["['t']", "[]"],
            "timestamp": ["2020", "2021"],
        }
    )

    records = build_chunk_records(df, chunk_size=512, overlap_tokens=102)

    assert records, "expected at least one record from the non-blank article"
    assert all(rec["metadata"]["title"] == "Real Title" for rec in records)
    first = records[0]
    assert first["id"] == "article_0_chunk_0"
    assert first["metadata"]["article_id"] == "0"
    assert first["metadata"]["authors"] == "Alice"
    assert first["metadata"]["text"] == first["text_for_embedding"]


def test_estimate_cost_counts_tokens_and_is_nonnegative() -> None:
    records = [{"text_for_embedding": "a short passage", "id": "x", "metadata": {}}]
    estimate = estimate_cost(records)

    assert estimate["chunks"] == 1.0
    assert estimate["tokens"] > 0
    assert estimate["usd"] >= 0
