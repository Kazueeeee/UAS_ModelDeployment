"""
ingest.py
---------
Handles raw data loading + structural cleaning (feature selection & dtype/junk fixing).
This is deliberately kept separate from `preprocessing.py`, which handles the
*statistical* transformations (imputation / encoding / scaling) that must be
fit on the train split only, to avoid data leakage.
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Abstract base class for all data ingestors."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def run(self) -> pd.DataFrame:
        df = self.load()
        df = self.clean(df)
        return df


class FeatureSelector:
    """Drops identifier / free-text / unused columns that carry no predictive signal
    or are unusable without heavy NLP-style feature engineering (e.g. Type_of_Loan)."""

    DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Type_of_Loan"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.DROP_COLS if c in df.columns]
        logger.info(f"Dropping unused columns: {cols}")
        return df.drop(columns=cols)


class DataCleaner:
    """Structural cleaning: fixes dtypes, replaces placeholder junk values with NaN,
    parses composite text fields into numeric, and clips values outside plausible
    domain ranges to NaN (to be statistically imputed later). All operations here
    are deterministic / row-independent, so they are safe to apply before the
    train/test split without causing data leakage."""

    NUMERIC_JUNK_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance",
    ]

    RANGE_LIMITS = {
        "Age": (0, 100),
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 40),
        "Num_of_Loan": (0, 15),
        "Num_of_Delayed_Payment": (0, 30),
        "Delay_from_due_date": (0, 120),
        "Num_Credit_Inquiries": (0, 30),
    }

    @staticmethod
    def _extract_numeric(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace("_", "", regex=False).str.strip()
        cleaned = cleaned.replace({"": np.nan, "nan": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")

    @staticmethod
    def _parse_credit_history_age(value):
        if pd.isna(value):
            return np.nan
        match = re.match(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(value))
        if not match:
            return np.nan
        years, months = int(match.group(1)), int(match.group(2))
        return years * 12 + months

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        obj_cols = df.select_dtypes(include="object").columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": np.nan, "": np.nan})

        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace("__10000__", np.nan)
        df["Monthly_Balance"] = df["Monthly_Balance"].replace(
            "__-333333333333333333333333333__", np.nan
        )

        for c in ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Month"]:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: x if pd.isna(x) else str(x).strip())

        for c in self.NUMERIC_JUNK_COLS:
            if c in df.columns:
                df[c] = self._extract_numeric(df[c])

        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(self._parse_credit_history_age)
            df = df.drop(columns=["Credit_History_Age"])

        for c, (low, high) in self.RANGE_LIMITS.items():
            if c in df.columns:
                df.loc[(df[c] < low) | (df[c] > high), c] = np.nan

        logger.info(f"Cleaned dataframe shape: {df.shape}")
        return df


class CreditScoreIngestor(BaseIngestor):
    """Concrete ingestor for the Overhead Credit Score dataset."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.feature_selector = FeatureSelector()
        self.cleaner = DataCleaner()

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Raw shape: {df.shape}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.feature_selector.transform(df)
        df = self.cleaner.transform(df)
        # drop rows without a target label (should be none, but keep pipeline defensive)
        df = df.dropna(subset=["Credit_Score"]).reset_index(drop=True)
        df = df.drop_duplicates().reset_index(drop=True)
        return df


if __name__ == "__main__":
    ingestor = CreditScoreIngestor("/home/claude/project/data/data_A.csv")
    df = ingestor.run()
    print(df.shape)
    print(df.isna().sum())
