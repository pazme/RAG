import sys
from pathlib import Path

import uvicorn

_SRC = Path(__file__).resolve().parents[1] / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from consts import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from rag import answer

app = FastAPI(title="Medium Article RAG Assistant")


class PromptRequest(BaseModel):
    question: str = Field(..., description="Question to answer.")


@app.get("/api/stats")
def stats() -> dict:
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }


@app.post("/api/prompt")
def prompt(request: PromptRequest) -> dict:
    return answer(request.question)


# for local run
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
