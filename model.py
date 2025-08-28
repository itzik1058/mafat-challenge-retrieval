from typing import TypedDict

import torch
from sentence_transformers import SentenceTransformer


class Paragraph(TypedDict):
    uuid: str
    passage: str


class Query(TypedDict):
    query_uuid: str
    query: str
    paragraphs: dict[str, Paragraph]
    target_actions: dict[str, str]
    case_name: str


class Prediction(TypedDict):
    paragraph_uuid: str
    score: float


class PreprocessedData(TypedDict):
    model: SentenceTransformer
    embeddings: dict[str, torch.Tensor]


def preprocess(corpus) -> PreprocessedData:
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embeddings = model.encode(
        [paragraph["passage"] for paragraph in corpus.values()],
        batch_size=16,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return {
        "model": model,
        "embeddings": dict(zip(corpus.keys(), embeddings.unbind())),
    }


def predict(query: Query, preprocessed_data: PreprocessedData) -> list[Prediction]:
    model = preprocessed_data["model"]
    embeddings = preprocessed_data["embeddings"]
    query_embedding = model.encode(
        query["query"],
        prompt_name="query",
        convert_to_tensor=True,
    )
    similarity = model.similarity(
        query_embedding,
        torch.stack(list(embeddings.values())),
    ).squeeze(0)
    return [
        {"paragraph_uuid": uuid, "score": score.item()}
        for uuid, score in zip(embeddings.keys(), similarity.unbind())
    ]
