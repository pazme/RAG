from pathlib import Path

import pandas as pd
import tiktoken
from pinecone import Pinecone

from consts import CHUNK_SIZE, OVERLAP_TOKENS, BATCH_SIZE, PINECONE_API_KEY, PINECONE_INDEX
from use_models import get_embeddings


def chunk_text(text: str, chunk_size: int, overlap_tokens: int):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        chunk_text_str = encoding.decode(chunk_tokens)
        chunks.append(chunk_text_str)

        start += (chunk_size - overlap_tokens)

    return chunks


def process_and_upload_articles(csv_path, articles_number=20):
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pinecone.Index(PINECONE_INDEX)

    print(f"Loading {articles_number} articles from {csv_path}")

    df = pd.read_csv(csv_path)
    df_sample = df.head(articles_number)

    all_chunks_metadata = []

    for index, row in df_sample.iterrows():
        text = str(row['text']) if pd.notna(row['text']) else ""
        title = str(row['title']) if pd.notna(row['title']) else "Unknown Title"

        if not text.strip():
            continue

        article_chunks = chunk_text(text, CHUNK_SIZE, OVERLAP_TOKENS)

        for chunk_idx, chunk_text_content in enumerate(article_chunks):
            metadata = {
                "article_id": str(index),
                "title": title,
                "authors": str(row.get('authors', '')),
                "url": str(row.get('url', '')),
                "text": chunk_text_content
            }

            chunk_id = f"article_{index}_chunk_{chunk_idx}"

            all_chunks_metadata.append({
                "id": chunk_id,
                "text_for_embedding": chunk_text_content,
                "metadata": metadata
            })

    print(f"Created a total of {len(all_chunks_metadata)} chunks. Starting batch upload...")

    # save embeddings to Pinecone
    for i in range(0, len(all_chunks_metadata), BATCH_SIZE):
        batch = all_chunks_metadata[i:i + BATCH_SIZE]

        texts_to_embed = [item["text_for_embedding"] for item in batch]

        embeddings = get_embeddings(texts_to_embed)

        vectors_to_upsert = []
        for j, embedding_vector in enumerate(embeddings):
            vectors_to_upsert.append({
                "id": batch[j]["id"],
                "values": embedding_vector,
                "metadata": batch[j]["metadata"]
            })

        pinecone_index.upsert(vectors=vectors_to_upsert)
        print(f"Successfully upserted chunks {i} to {i + len(batch)}...")

    print("Pipeline finished successfully! Data is ready for querying.")


csv_file_path = Path(__file__).parent / "data" / "medium-english-50mb.csv"
process_and_upload_articles(csv_file_path, articles_number=20)
