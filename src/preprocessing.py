"""
preprocessing.py
-----------------
Handles the *statistical* preprocessing steps that must be fit on the training
split only: train/test split, imputation, encoding, and scaling. Numeric columns
are routed to a "skewed" branch (median imputation + RobustScaler) or a "normal"
branch (mean imputation + StandardScaler) based on the skewness measured on the
training data. Categorical columns are imputed with the most frequent value and
one-hot encoded.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame):
        ...

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)


class DataSplitter:
    """Stratified train/test splitter."""

    def __init__(self, target: str, test_size: float = 0.2, random_state: int = 42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target])
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test


class FeaturePreprocessor(BasePreprocessor):
    """Skew-aware imputation/scaling for numeric columns + OHE for categoricals."""

    SKEW_THRESHOLD = 0.5

    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None
        self.skewed_cols = None
        self.normal_cols = None
        self.pipeline: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X.select_dtypes(include="object").columns.tolist()

        skew = X[self.numeric_cols].skew()
        self.skewed_cols = skew[skew.abs() > self.SKEW_THRESHOLD].index.tolist()
        self.normal_cols = [c for c in self.numeric_cols if c not in self.skewed_cols]
        logger.info(f"Skewed numeric cols (median+RobustScaler): {self.skewed_cols}")
        logger.info(f"~Normal numeric cols (mean+StandardScaler): {self.normal_cols}")

        skewed_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        normal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers = []
        if self.skewed_cols:
            transformers.append(("skewed", skewed_pipe, self.skewed_cols))
        if self.normal_cols:
            transformers.append(("normal", normal_pipe, self.normal_cols))
        if self.categorical_cols:
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.pipeline = ColumnTransformer(transformers, remainder="drop")
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        return self.pipeline.transform(X)

    def get_feature_names(self):
        return self.pipeline.get_feature_names_out()

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str) -> "FeaturePreprocessor":
        return joblib.load(path)


class TargetEncoder:
    """Wraps sklearn LabelEncoder for the multi-class target column."""

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit_transform(self, y):
        return self.encoder.fit_transform(y)

    def transform(self, y):
        return self.encoder.transform(y)

    def inverse_transform(self, y):
        return self.encoder.inverse_transform(y)

    @property
    def classes_(self):
        return self.encoder.classes_

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TargetEncoder":
        return joblib.load(path)


if __name__ == "__main__":
    from ingest import CreditScoreIngestor

    df = CreditScoreIngestor("/home/claude/project/data/data_A.csv").run()
    X_train, X_test, y_train, y_test = DataSplitter("Credit_Score").split(df)

    fp = FeaturePreprocessor()
    Xtr = fp.fit_transform(X_train)
    Xte = fp.transform(X_test)
    print(Xtr.shape, Xte.shape)

    te = TargetEncoder()
    ytr = te.fit_transform(y_train)
    yte = te.transform(y_test)
    print(te.classes_)
