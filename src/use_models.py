from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from consts import (
    EMBEDDING_MODEL,
    GPT_5_MODEL,
    LLMOD_AI_API_KEY,
    LLMOD_AI_URL,
)


def _embeddings_client(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=LLMOD_AI_API_KEY,
        base_url=LLMOD_AI_URL,
        model=model,
    )


def get_embeddings(texts: List[str]) -> List[List[float]]:
    # one call for the whole batch
    return _embeddings_client().embed_documents(texts)


def embed_query(text: str) -> List[float]:
    return _embeddings_client().embed_query(text)


def get_gpt5(system_prompt: str, user_prompt: str) -> str:
    llm = ChatOpenAI(
        api_key=LLMOD_AI_API_KEY,
        base_url=LLMOD_AI_URL,
        model=GPT_5_MODEL,
    )

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    return response.content
