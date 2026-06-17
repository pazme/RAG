import logging
from typing import Dict, List

from pinecone import Pinecone

from consts import PINECONE_API_KEY, PINECONE_INDEX, TOP_K
from use_models import embed_query, get_gpt5

logger = logging.getLogger(__name__)

# required grounding prompt (per assignment) + a short style note
SYSTEM_PROMPT: str = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. If the "
    "answer cannot be determined from the provided context, respond: \"I don't know "
    "based on the provided Medium articles data.\" Always explain your answer using "
    "the given context, quoting or paraphrasing the relevant article passage or "
    "metadata when helpful.\n\n"
    "Response style: be concise and direct. When asked for specific fields (e.g. "
    "title and author) return exactly those. When asked to list N articles, return N "
    "distinct articles (never multiple passages of the same article)."
)

_pinecone_index = None


def _get_index():
    global _pinecone_index

    if _pinecone_index is None:
        _pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)

    return _pinecone_index


def retrieve(
    question: str,
    top_k: int = TOP_K,
    namespace: str = "",
) -> List[Dict[str, object]]:
    query_vector = embed_query(question)
    result = _get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    contexts: List[Dict[str, object]] = []

    for match in result.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        contexts.append(
            {
                "article_id": metadata.get("article_id", ""),
                "title": metadata.get("title", ""),
                "chunk": metadata.get("text", ""),
                "score": round(float(match.get("score", 0.0)), 4),
                "url": metadata.get("url", ""),
                "authors": metadata.get("authors", ""),
            }
        )

    return contexts


def build_user_prompt(question: str, contexts: List[Dict[str, object]]) -> str:
    if not contexts:
        return f"No article passages were retrieved.\n\nQuestion: {question}"

    # label each passage so the model can tell articles apart (needed for "list N")
    blocks: List[str] = []

    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"[Passage {i}]\n"
            f"article_id: {ctx['article_id']}\n"
            f"title: {ctx['title']}\n"
            f"author: {ctx['authors']}\n"
            f"url: {ctx['url']}\n"
            f"text: {ctx['chunk']}"
        )

    context_block = "\n\n".join(blocks)

    return (
        "Use only the following retrieved Medium article passages to answer.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}"
    )


def answer(
    question: str,
    top_k: int = TOP_K,
    namespace: str = "",
) -> Dict[str, object]:
    contexts = retrieve(question, top_k=top_k, namespace=namespace)
    user_prompt = build_user_prompt(question, contexts)

    response_text = get_gpt5(SYSTEM_PROMPT, user_prompt)

    return {
        "response": response_text,
        "context": [
            {
                "article_id": ctx["article_id"],
                "title": ctx["title"],
                "chunk": ctx["chunk"],
                "score": ctx["score"],
            }
            for ctx in contexts
        ],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
