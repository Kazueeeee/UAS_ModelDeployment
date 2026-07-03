"""
training.py
------------
OOP training layer. `BaseModelTrainer` is an abstract class that defines the
common training contract (fit / predict / predict_proba / save). Each concrete
tree-based model (RandomForest, XGBoost, GradientBoosting, ...) is implemented
as a subclass that only needs to define `build_model()`. This demonstrates
both abstraction (ABC) and inheritance, and lets `pipeline.py` orchestrate all
trainers polymorphically through the same interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """Abstract base trainer. Subclasses only need to implement `build_model`."""

    name: str = "base"

    def __init__(self, random_state: int = 42, **model_kwargs):
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        """Return an unfitted sklearn-compatible estimator instance."""
        ...

    def fit(self, X_train, y_train):
        logger.info(f"[{self.name}] training on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] training complete.")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params()

    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"[{self.name}] model saved to {path}")

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class RandomForestTrainer(BaseModelTrainer):
    name = "RandomForest"

    def build_model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self.model_kwargs.get("n_estimators", 200),
            max_depth=self.model_kwargs.get("max_depth", 16),
            random_state=self.random_state,
            n_jobs=-1,
        )


class GradientBoostingTrainer(BaseModelTrainer):
    name = "GradientBoosting"

    def build_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=self.model_kwargs.get("n_estimators", 100),
            learning_rate=self.model_kwargs.get("learning_rate", 0.1),
            random_state=self.random_state,
        )


class XGBoostTrainer(BaseModelTrainer):
    name = "XGBoost"

    def build_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=self.model_kwargs.get("n_estimators", 300),
            max_depth=self.model_kwargs.get("max_depth", 6),
            learning_rate=self.model_kwargs.get("learning_rate", 0.1),
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )


class ExtraTreesTrainer(BaseModelTrainer):
    name = "ExtraTrees"

    def build_model(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=self.model_kwargs.get("n_estimators", 200),
            max_depth=self.model_kwargs.get("max_depth", 16),
            random_state=self.random_state,
            n_jobs=-1,
        )


class DecisionTreeTrainer(BaseModelTrainer):
    name = "DecisionTree"

    def build_model(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            max_depth=self.model_kwargs.get("max_depth", 12),
            random_state=self.random_state,
        )


TRAINER_REGISTRY = {
    "RandomForest": RandomForestTrainer,
    "GradientBoosting": GradientBoostingTrainer,
    "XGBoost": XGBoostTrainer,
    "ExtraTrees": ExtraTreesTrainer,
    "DecisionTree": DecisionTreeTrainer,
}
