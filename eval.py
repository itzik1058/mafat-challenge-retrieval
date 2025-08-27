import json
from pathlib import Path

from sklearn.metrics import ndcg_score
from typer import Typer

from model import predict, preprocess

app = Typer()


@app.command()
def main(limit_queries: int | None = None):
    data_path = Path("env/hsrc")
    # cache_path = Path("env/preprocessed_data.json")

    with (data_path / "hsrc_corpus.jsonl").open() as f:
        corpus_entries = [json.loads(line) for line in f.readlines()]
    corpus = {entry["uuid"]: entry for entry in corpus_entries}

    with (data_path / "hsrc_train.jsonl").open() as f:
        train_queries = [json.loads(line) for line in f.readlines()]

    if limit_queries is not None:
        train_queries = train_queries[:limit_queries]
        # train_paragraphs = [
        #     paragraph["uuid"]
        #     for query in train_queries
        #     for paragraph in query["paragraphs"].values()
        # ]
        # corpus = {
        #     uuid: entry for uuid, entry in corpus.items() if uuid in train_paragraphs
        # }
    print(f"Evaluating on {len(train_queries)} queries with {len(corpus)} paragraphs.")

    preprocessed_data = preprocess(corpus)

    predictions = [predict(query, preprocessed_data) for query in train_queries]

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
            scores.append(predicted_scores[uuid])
        ndcg_20.append(ndcg_score([true_relevance], [scores], k=20))
    mean_ndcg_20 = sum(ndcg_20) / len(ndcg_20)

    print(mean_ndcg_20)


if __name__ == "__main__":
    app()
