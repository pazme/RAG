from typing import List

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from consts import LLMOD_AI_API_KEY, LLMOD_AI_URL, GPT_5_MODEL, EMBEDDING_MODEL


def get_gpt5(prompt: str) -> AIMessage:
    llm = ChatOpenAI(
        api_key=LLMOD_AI_API_KEY,
        base_url=LLMOD_AI_URL,
        model=GPT_5_MODEL
    )

    response = llm.invoke(prompt)
    return response


def get_embeddings(text: List[str]) -> List[List[float]]:
    embeddings = OpenAIEmbeddings(
        api_key=LLMOD_AI_API_KEY,
        base_url=LLMOD_AI_URL,
        model=EMBEDDING_MODEL
    )

    vectors = embeddings.embed_documents(text)
    return vectors
