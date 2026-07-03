"""
pipeline.py
-----------
Orchestrates the full local training pipeline:

    ingest -> split -> preprocess -> train (3 candidate models) -> evaluate
    -> log everything to MLflow -> pick the best model -> persist .pkl artifacts

Run directly:
    python pipeline.py

Then inspect results with:
    mlflow ui --backend-store-uri ./mlruns
"""

from __future__ import annotations

import os
import logging

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from ingest import CreditScoreIngestor
from preprocessing import DataSplitter, FeaturePreprocessor, TargetEncoder
from training import TRAINER_REGISTRY
from evaluation import ModelEvaluator, ModelComparator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data_A.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
TARGET = "Credit_Score"

# The 3 best-performing tree models found during experimentation in eksperiment.ipynb
CANDIDATE_MODELS = ["RandomForest", "XGBoost", "GradientBoosting"]


class TrainingPipeline:
    """Orchestrates ingestion, preprocessing, training, evaluation and MLflow logging."""

    def __init__(
        self,
        data_path: str = DATA_PATH,
        artifact_dir: str = ARTIFACT_DIR,
        candidate_models: list[str] | None = None,
        experiment_name: str = "credit_score_classification",
    ):
        self.data_path = data_path
        self.artifact_dir = artifact_dir
        self.candidate_models = candidate_models or CANDIDATE_MODELS
        self.experiment_name = experiment_name

        os.makedirs(self.artifact_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        mlflow.set_experiment(self.experiment_name)

        self.preprocessor = FeaturePreprocessor()
        self.target_encoder = TargetEncoder()
        self.comparator = ModelComparator(metric="f1_macro")
        self.trainers = {}

    # -- steps -------------------------------------------------------
    def ingest(self):
        logger.info("STEP 1/5: Ingest & clean data")
        self.df = CreditScoreIngestor(self.data_path).run()
        return self.df

    def split_and_preprocess(self):
        logger.info("STEP 2/5: Split & preprocess")
        self.X_train, self.X_test, self.y_train_raw, self.y_test_raw = DataSplitter(TARGET).split(self.df)

        self.X_train_t = self.preprocessor.fit_transform(self.X_train)
        self.X_test_t = self.preprocessor.transform(self.X_test)

        self.y_train = self.target_encoder.fit_transform(self.y_train_raw)
        self.y_test = self.target_encoder.transform(self.y_test_raw)
        return self.X_train_t, self.X_test_t, self.y_train, self.y_test

    def train_and_evaluate(self):
        logger.info("STEP 3/5: Train & evaluate candidate models")
        with mlflow.start_run(run_name="training_pipeline") as parent_run:
            mlflow.log_param("candidate_models", self.candidate_models)
            mlflow.log_param("n_train", self.X_train_t.shape[0])
            mlflow.log_param("n_test", self.X_test_t.shape[0])
            mlflow.log_param("n_features", self.X_train_t.shape[1])
            mlflow.log_param("skewed_cols", self.preprocessor.skewed_cols)
            mlflow.log_param("normal_cols", self.preprocessor.normal_cols)

            for name in self.candidate_models:
                trainer_cls = TRAINER_REGISTRY[name]
                with mlflow.start_run(run_name=name, nested=True):
                    trainer = trainer_cls(random_state=42)
                    trainer.fit(self.X_train_t, self.y_train)
                    self.trainers[name] = trainer

                    evaluator = ModelEvaluator(
                        trainer, self.X_test_t, self.y_test, class_names=self.target_encoder.classes_
                    )
                    metrics = evaluator.evaluate()
                    self.comparator.add_result(metrics)

                    mlflow.log_params(trainer.get_params())
                    mlflow.log_metrics({
                        "accuracy": metrics["accuracy"],
                        "precision_macro": metrics["precision_macro"],
                        "recall_macro": metrics["recall_macro"],
                        "f1_macro": metrics["f1_macro"],
                    })
                    if name == "XGBoost":
                        mlflow.xgboost.log_model(trainer.model, name=f"model_{name}")
                    else:
                        mlflow.sklearn.log_model(trainer.model, name=f"model_{name}")

                    report_path = os.path.join(self.artifact_dir, f"classification_report_{name}.txt")
                    with open(report_path, "w") as f:
                        f.write(evaluator.classification_report())
                    mlflow.log_artifact(report_path)

            results_df = self.comparator.to_frame()
            logger.info(f"\n{results_df}")
            results_path = os.path.join(self.artifact_dir, "model_comparison.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)

        return self.comparator

    def select_best_and_persist(self):
        logger.info("STEP 4/5: Select best model")
        best_name = self.comparator.best_model_name()
        best_trainer = self.trainers[best_name]
        logger.info(f"Best model: {best_name}")

        logger.info("STEP 5/5: Persist .pkl artifacts")
        best_trainer.save(os.path.join(self.artifact_dir, "best_model.pkl"))
        self.preprocessor.save(os.path.join(self.artifact_dir, "preprocessor.pkl"))
        self.target_encoder.save(os.path.join(self.artifact_dir, "target_encoder.pkl"))

        with open(os.path.join(self.artifact_dir, "best_model_name.txt"), "w") as f:
            f.write(best_name)

        return best_name, best_trainer

    def run(self):
        self.ingest()
        self.split_and_preprocess()
        self.train_and_evaluate()
        best_name, best_trainer = self.select_best_and_persist()
        logger.info(f"Pipeline complete. Best model: {best_name}. Artifacts in: {self.artifact_dir}")
        return best_name, best_trainer, self.comparator.to_frame()


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    best_name, best_trainer, results = pipeline.run()
    print("\n=== FINAL RESULTS ===")
    print(results)
    print(f"\nBest model: {best_name}")
