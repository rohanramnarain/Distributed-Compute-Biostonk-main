"""Train a browser-compatible trial outcome model from outcome-labelled CSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_examples(path: Path, text_column: str, outcome_column: str) -> tuple[list[str], list[int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    examples = [(row.get(text_column, "").strip(), row.get(outcome_column, "").strip()) for row in rows]
    if any(not text or outcome not in {"0", "1"} for text, outcome in examples):
        raise ValueError("Every CSV row must contain non-empty text and a literal binary 0 or 1 outcome.")
    texts = [text for text, _ in examples]
    outcomes = [int(outcome) for _, outcome in examples]
    if set(outcomes) != {0, 1}:
        raise ValueError("The CSV must include examples for both binary outcome classes.")
    return texts, outcomes


def train(input_path: Path, output_path: Path, text_column: str, outcome_column: str) -> None:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError("Install clinical/requirements.txt before training an outcome model.") from error
    texts, outcomes = read_examples(input_path, text_column, outcome_column)
    vectorizer = HashingVectorizer(
        n_features=16384,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(vectorizer.transform(texts), outcomes)
    weights = {
        str(index): float(weight)
        for index, weight in enumerate(classifier.coef_[0])
        if abs(weight) >= 1e-9
    }
    artifact = {
        "version": "outcome-model-1",
        "model_type": "trial_success_probability",
        "n_features": 16384,
        "calibrated_for_outcomes": False,
        "training_note": "Requires temporal holdout calibration and external validation before clinical use.",
        "hashing": {
            "implementation": "MurmurHash3 32-bit, seed 0; compatible with trial-scorer.js",
            "token_pattern": "(?u)\\b\\w\\w+\\b",
            "alternate_sign": False
        },
        "baselines": {"default": {"base_logit": float(classifier.intercept_[0]), "similarity_score": None}},
        "sparse_weights": weights,
        "phrase_weights": []
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact browser trial-outcome model.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("clinical/static/models/trial-outcome-model.json"))
    parser.add_argument("--text-column", default="protocol_text")
    parser.add_argument("--outcome-column", default="success")
    arguments = parser.parse_args()
    train(arguments.input_csv, arguments.output, arguments.text_column, arguments.outcome_column)


if __name__ == "__main__":
    main()