"""Train and export a browser-compatible completed-versus-terminated classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


MODEL_TYPE = "trial_completion_vs_termination"
DEFAULT_BUCKET_COUNT = 32


def nct_bucket(nct_id: str, bucket_count: int) -> int:
    """Return the stable bucket shared by the artifact writer and browser loader."""
    digest = hashlib.sha256(nct_id.upper().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % bucket_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[list[str], list[list[float]], list[int], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        embedding_columns = [column for column in columns if column.startswith("emb_")]
        if not embedding_columns:
            raise ValueError("The CSV must include emb_0 through emb_N columns.")
        expected_columns = [f"emb_{index}" for index in range(len(embedding_columns))]
        if embedding_columns != expected_columns:
            raise ValueError("Embedding columns must be contiguous and ordered emb_0 through emb_N.")
        required_columns = {"nct_id", "success"}
        missing_columns = sorted(required_columns.difference(columns))
        if missing_columns:
            raise ValueError(f"The CSV is missing required columns: {', '.join(missing_columns)}.")

        nct_ids: list[str] = []
        embeddings: list[list[float]] = []
        labels: list[int] = []
        seen_nct_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            nct_id = row["nct_id"].strip().upper()
            label = row["success"].strip()
            if not nct_id:
                raise ValueError(f"Row {row_number} has an empty nct_id.")
            if nct_id in seen_nct_ids:
                raise ValueError(f"Row {row_number} duplicates nct_id {nct_id}.")
            if label not in {"0", "1"}:
                raise ValueError(f"Row {row_number} has a non-binary success label.")
            try:
                embedding = [float(row[column]) for column in embedding_columns]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number} has a non-numeric embedding value.") from error
            if not all(float("-inf") < value < float("inf") for value in embedding):
                raise ValueError(f"Row {row_number} has a non-finite embedding value.")
            seen_nct_ids.add(nct_id)
            nct_ids.append(nct_id)
            embeddings.append(embedding)
            labels.append(int(label))

    if not nct_ids:
        raise ValueError("The CSV contains no data rows.")
    if set(labels) != {0, 1}:
        raise ValueError("The CSV must include both completed (1) and terminated (0) labels.")
    return nct_ids, embeddings, labels, embedding_columns


def split_metrics(y_true: Any, completed_probability: Any, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    terminated_true = 1 - y_true
    terminated_probability = 1 - completed_probability
    terminated_prediction = (terminated_probability >= (1 - threshold)).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        terminated_true,
        terminated_prediction,
        average=None,
        labels=[1],
        zero_division=0,
    )
    return {
        "completed_roc_auc": float(roc_auc_score(y_true, completed_probability)),
        "terminated_average_precision": float(average_precision_score(terminated_true, terminated_probability)),
        "completed_brier_score": float(brier_score_loss(y_true, completed_probability)),
        "termination_threshold": float(1 - threshold),
        "terminated_precision": float(precision[0]),
        "terminated_recall": float(recall[0]),
        "terminated_f1": float(f1[0]),
        "terminated_support": int(support[0]),
        "confusion_matrix_terminated_positive": confusion_matrix(
            terminated_true, terminated_prediction, labels=[0, 1]
        ).tolist(),
    }


def choose_completion_threshold(y_true: Any, completed_probability: Any) -> float:
    from sklearn.metrics import precision_recall_curve

    terminated_true = 1 - y_true
    precision, recall, thresholds = precision_recall_curve(terminated_true, 1 - completed_probability)
    if not len(thresholds):
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    termination_threshold = float(thresholds[int(f1_scores.argmax())])
    return 1 - termination_threshold


def write_embedding_buckets(
    nct_ids: list[str], embeddings: list[list[float]], output_directory: Path, bucket_count: int
) -> None:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for nct_id, embedding in zip(nct_ids, embeddings, strict=True):
        buckets[nct_bucket(nct_id, bucket_count)].append({"nct_id": nct_id, "embedding": embedding})
    output_directory.mkdir(parents=True, exist_ok=True)
    for bucket_number in range(bucket_count):
        payload = {
            "version": "trial-completion-v1",
            "embedding_dim": len(embeddings[0]),
            "bucket": bucket_number,
            "bucket_count": bucket_count,
            "records": buckets[bucket_number],
        }
        target = output_directory / f"bucket-{bucket_number:02d}.json"
        target.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def train(
    input_path: Path,
    model_output: Path,
    embedding_output_directory: Path,
    bucket_count: int,
    random_state: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("Install clinical/requirements.txt before training an embedding classifier.") from error

    nct_ids, embeddings, labels, embedding_columns = read_dataset(input_path)
    features = np.asarray(embeddings, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.int8)
    indices = np.arange(len(outcomes))
    train_indices, holdout_indices = train_test_split(
        indices, test_size=0.3, stratify=outcomes, random_state=random_state
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.5,
        stratify=outcomes[holdout_indices],
        random_state=random_state,
    )
    scaler = StandardScaler().fit(features[train_indices])
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    classifier.fit(scaler.transform(features[train_indices]), outcomes[train_indices])

    validation_probability = classifier.predict_proba(scaler.transform(features[validation_indices]))[:, 1]
    completion_threshold = choose_completion_threshold(outcomes[validation_indices], validation_probability)
    test_probability = classifier.predict_proba(scaler.transform(features[test_indices]))[:, 1]
    class_counts = Counter(int(label) for label in outcomes)
    model = {
        "version": "trial-completion-v1",
        "model_type": MODEL_TYPE,
        "label_definition": {"0": "terminated", "1": "completed"},
        "embedding_dim": len(embedding_columns),
        "embedding_columns": embedding_columns,
        "standardization": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "weights": classifier.coef_[0].tolist(),
        "base_logit": float(classifier.intercept_[0]),
        "classification": {"completion_threshold": completion_threshold},
        "calibrated_for_outcomes": False,
        "training_note": "Class-weighted logistic regression on supplied embeddings. The completion score is not externally validated or clinically calibrated.",
        "source": {"filename": input_path.name, "sha256": sha256_file(input_path)},
        "training": {
            "created_on": date.today().isoformat(),
            "random_state": random_state,
            "class_weight": "balanced",
            "sample_count": len(outcomes),
            "class_counts": {"terminated": class_counts[0], "completed": class_counts[1]},
            "split_counts": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
        },
        "validation_metrics": split_metrics(outcomes[validation_indices], validation_probability, completion_threshold),
        "test_metrics": split_metrics(outcomes[test_indices], test_probability, completion_threshold),
        "embedding_lookup": {
            "bucket_count": bucket_count,
            "path_template": "/static/models/trial-embeddings/bucket-{bucket:02d}.json",
            "record_count": len(nct_ids),
        },
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    write_embedding_buckets(nct_ids, embeddings, embedding_output_directory, bucket_count)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a browser-compatible completion-versus-termination classifier.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model-output", type=Path, default=Path("clinical/static/models/trial-completion-model.json"))
    parser.add_argument("--embedding-output-directory", type=Path, default=Path("clinical/static/models/trial-embeddings"))
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT)
    parser.add_argument("--random-state", type=int, default=20260811)
    arguments = parser.parse_args()
    if arguments.bucket_count < 1:
        parser.error("--bucket-count must be at least 1")
    model = train(
        arguments.input_csv,
        arguments.model_output,
        arguments.embedding_output_directory,
        arguments.bucket_count,
        arguments.random_state,
    )
    metrics = model["test_metrics"]
    print(
        "Trained completion-versus-termination classifier "
        f"on {model['training']['sample_count']} NCTs; "
        f"terminated PR-AUC={metrics['terminated_average_precision']:.3f}, "
        f"completed ROC-AUC={metrics['completed_roc_auc']:.3f}."
    )


if __name__ == "__main__":
    main()