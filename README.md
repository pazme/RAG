# Medium Article RAG Assistant

RAG over ~7,600 Medium articles. A question gets embedded, the closest chunks are
pulled from Pinecone, and gpt-5-mini answers using only those chunks (no outside
knowledge). Embeddings + chat both go through the LLMOD gateway.

## Layout

```
api/index.py        FastAPI app -> /api/stats, /api/prompt
src/consts.py       config + hyperparams (secrets from env)
src/use_models.py   gateway wrappers: get_embeddings, embed_query, get_gpt5
src/rag.py          retrieve + build prompt + answer
src/ingest.py       chunk -> embed -> upsert (run once)
src/experiment.py   chunk-size comparison on a subset
```

## Setup

```powershell
copy .env.example .env   # fill in the keys
.venv\Scripts\python.exe -m pip install -r requirements-ingest.txt
```

Env vars: `LLMOD_AI_API_KEY`, `LLMOD_AI_URL`, `PINECONE_API_KEY`, `PINECONE_INDEX`.

## Ingest

Nothing embeds on import — only when you run the script. Check cost first with
`--estimate`.

```powershell
.venv\Scripts\python.exe src\ingest.py --articles 50 --estimate
.venv\Scripts\python.exe src\ingest.py --articles 7682
```

Full corpus is ~29k chunks, ~$0.25 to embed. Do it once; chunks go up 50 at a time.

## Hyperparameters

| param         | value |
|---------------|-------|
| chunk_size    | 512   |
| overlap_ratio | 0.2   |
| top_k         | 12    |

Picked 512 after comparing chunk sizes on a 200-article subset (known-item recall):

| chunk_size | recall@1 | recall@5 | recall@10 | mrr   |
|------------|----------|----------|-----------|-------|
| 256        | 0.960    | 1.000    | 1.000     | 0.973 |
| 512        | 0.960    | 0.960    | 0.960     | 0.960 |
| 1024       | 0.720    | 0.880    | 0.960     | 0.785 |

1024 was clearly worse. 256 and 512 are basically tied, but 512 gives longer
passages (better for the summary/recommend questions) and half the vectors, so I
went with 512. Rerun with `src\experiment.py`.

## API

`GET /api/stats`

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 12
}
```

`POST /api/prompt`

```json
{
  "question": "List exactly 3 articles about education. Return only the titles."
}
```

returns `response`, `context` (article_id/title/chunk/score) and `Augmented_prompt`
(System/User). If the answer isn't in the context it replies exactly:
`I don't know based on the provided Medium articles data.`

## Tests

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```
