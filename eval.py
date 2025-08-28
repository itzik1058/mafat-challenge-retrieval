import json
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from typer import Typer

from model import Paragraph, Query, predict, preprocess

app = Typer()


@app.command()
def main(limit_queries: int | None = None):
    data_path = Path("env/hsrc")
    # cache_path = Path("env/preprocessed_data.json")

    with (data_path / "hsrc_corpus.jsonl").open() as f:
        paragraphs = [json.loads(line) for line in f.readlines()]
    corpus: dict[str, Paragraph] = {
        paragraph["uuid"]: paragraph for paragraph in paragraphs
    }

    with (data_path / "hsrc_train.jsonl").open() as f:
        train_queries: list[Query] = [json.loads(line) for line in f.readlines()]

    if limit_queries is not None:
        train_queries = train_queries[:limit_queries]
        train_paragraphs = [
            paragraph["uuid"]
            for query in train_queries
            for paragraph in query["paragraphs"].values()
        ]
        corpus = {
            uuid: entry for uuid, entry in corpus.items() if uuid in train_paragraphs
        }
    print(f"Evaluating on {len(train_queries)} queries with {len(corpus)} paragraphs.")

    preprocessed_data = preprocess(corpus)

    progress = tqdm(train_queries)
    predictions = (predict(query, preprocessed_data) for query in progress)

    ndcg_20 = []
    for query, prediction in zip(train_queries, predictions):
        relevant_paragraph_mapping = {
            tag.lstrip("paragraph_"): p["uuid"]
            for tag, p in query["paragraphs"].items()
        }
        paragraph_relevance = {
            relevant_paragraph_mapping[tag.lstrip("target_action_")]: int(score)
            for tag, score in query["target_actions"].items()
        }
        predicted_scores = {p["paragraph_uuid"]: p["score"] for p in prediction}

        true_relevance, scores = [], []
        for uuid, score in paragraph_relevance.items():
            true_relevance.append(score)
            scores.append(predicted_scores.get(uuid, -1000))
        ndcg_20.append(ndcg_score([true_relevance], [scores], k=20))

        progress.set_postfix_str(f"NDCG@20: {np.mean(ndcg_20):.4f}")

    print(np.mean(ndcg_20))


if __name__ == "__main__":
    app()
