"""
evaluation.py
-------------
Computes classification metrics for a trained model and produces a
comparison table across multiple trained models so the best one can be
selected.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a single fitted trainer against a held-out test set."""

    def __init__(self, trainer, X_test, y_test, class_names=None):
        self.trainer = trainer
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.y_pred = None

    def evaluate(self) -> dict:
        self.y_pred = self.trainer.predict(self.X_test)
        metrics = {
            "model": self.trainer.name,
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision_macro": precision_score(self.y_test, self.y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(self.y_test, self.y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(self.y_test, self.y_pred, average="macro", zero_division=0),
        }
        logger.info(f"[{self.trainer.name}] {metrics}")
        return metrics

    def confusion_matrix(self):
        if self.y_pred is None:
            self.y_pred = self.trainer.predict(self.X_test)
        return confusion_matrix(self.y_test, self.y_pred)

    def classification_report(self):
        if self.y_pred is None:
            self.y_pred = self.trainer.predict(self.X_test)
        target_names = [str(c) for c in self.class_names] if self.class_names is not None else None
        return classification_report(self.y_test, self.y_pred, target_names=target_names, zero_division=0)


class ModelComparator:
    """Aggregates evaluation results from several trainers into one comparison table
    and selects the best model by a chosen metric."""

    def __init__(self, metric: str = "f1_macro"):
        self.metric = metric
        self.results: list[dict] = []

    def add_result(self, metrics: dict):
        self.results.append(metrics)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.results).sort_values(self.metric, ascending=False).reset_index(drop=True)

    def best_model_name(self) -> str:
        df = self.to_frame()
        return df.iloc[0]["model"]
