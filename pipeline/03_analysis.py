"""
03_analysis.py

Queries the MongoDB patient-centered collection, creates an encounter-level
analysis dataframe, trains readmission prediction models, and saves model
outputs for visualization and reporting.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DB_NAME = "hospital_readmissions"
COLLECTION_NAME = "patients"


def setup_logging() -> None:
    """Configure logging to both a file and the console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_DIR / "analysis.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def get_mongo_collection():
    """Connect to MongoDB and return the patient collection."""
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        raise ValueError("MONGO_URI not found. Add it to your .env file.")

    try:
        client = MongoClient(mongo_uri)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        logging.info("Connected to MongoDB database: %s", DB_NAME)
        logging.info("Collection document count: %s", collection.count_documents({}))

        return collection

    except PyMongoError as e:
        logging.error("MongoDB connection error: %s", e)
        raise


def flatten_patient_documents(collection) -> pd.DataFrame:
    """
    Convert patient-centered MongoDB documents into an encounter-level dataframe.

    Each row represents one encounter, with demographic fields from the patient
    document and clinical complexity measures from embedded encounter arrays.
    """
    records = []

    patients = collection.find({"_id": {"$ne": "database_metadata"}})

    for patient in patients:
        patient_id = patient.get("Id")
        birthdate = patient.get("BIRTHDATE")
        gender = patient.get("GENDER")
        race = patient.get("RACE")
        ethnicity = patient.get("ETHNICITY")

        for encounter in patient.get("encounters", []):
            records.append(
                {
                    "patient_id": patient_id,
                    "encounter_id": encounter.get("Id"),
                    "birthdate": birthdate,
                    "gender": gender,
                    "race": race,
                    "ethnicity": ethnicity,
                    "encounter_class": encounter.get("ENCOUNTERCLASS"),
                    "encounter_description": encounter.get("DESCRIPTION"),
                    "start": encounter.get("START"),
                    "stop": encounter.get("STOP"),
                    "num_conditions": len(encounter.get("conditions", [])),
                    "num_medications": len(encounter.get("medications", [])),
                    "num_procedures": len(encounter.get("procedures", [])),
                    "num_observations": len(encounter.get("observations", [])),
                    "readmitted_30_days": encounter.get("readmitted_30_days", False),
                }
            )

    df = pd.DataFrame(records)
    logging.info("Flattened MongoDB documents into %s encounter rows", len(df))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from the encounter-level dataframe."""
    df = df.copy()

    df["birthdate"] = pd.to_datetime(df["birthdate"], errors="coerce", utc=True)
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df["stop"] = pd.to_datetime(df["stop"], errors="coerce", utc=True)

    df["age_at_encounter"] = (df["start"] - df["birthdate"]).dt.days / 365.25

    df["length_of_stay_days"] = (
        (df["stop"] - df["start"]).dt.total_seconds() / (60 * 60 * 24)
    )

    df["readmitted_30_days"] = df["readmitted_30_days"].astype(int)

    df = df.dropna(
        subset=[
            "age_at_encounter",
            "length_of_stay_days",
            "readmitted_30_days",
        ]
    )

    df = df[df["length_of_stay_days"] >= 0]

    df = df.sort_values(["patient_id", "start"])
    df["prior_encounters"] = df.groupby("patient_id").cumcount()

    logging.info("Completed feature engineering")
    logging.info("Final analysis dataframe rows: %s", len(df))
    logging.info("Readmission rate: %.4f", df["readmitted_30_days"].mean())

    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    """Build preprocessing steps for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def evaluate_model(model_name: str, y_test, predictions, probabilities) -> dict:
    """Calculate model evaluation metrics."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train logistic regression and random forest models.

    Saves model comparison metrics, test-set predictions, and random forest
    feature importance values.
    """
    features = [
        "age_at_encounter",
        "length_of_stay_days",
        "prior_encounters",
        "num_conditions",
        "num_medications",
        "num_procedures",
        "num_observations",
        "gender",
        "race",
        "ethnicity",
        "encounter_class",
    ]

    target = "readmitted_30_days"

    numeric_features = [
        "age_at_encounter",
        "length_of_stay_days",
        "prior_encounters",
        "num_conditions",
        "num_medications",
        "num_procedures",
        "num_observations",
    ]

    categorical_features = [
        "gender",
        "race",
        "ethnicity",
        "encounter_class",
    ]

    modeling_df = df.dropna(subset=features + [target]).copy()

    X = modeling_df[features]
    y = modeling_df[target]

    if y.nunique() < 2:
        raise ValueError("Target variable has only one class. Cannot train classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    max_depth=8,
                ),
            ),
        ]
    )

    logging.info("Training logistic regression model")
    logistic_model.fit(X_train, y_train)

    logging.info("Training random forest model")
    rf_model.fit(X_train, y_train)

    logistic_preds = logistic_model.predict(X_test)
    logistic_probs = logistic_model.predict_proba(X_test)[:, 1]

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    model_results = pd.DataFrame(
        [
            evaluate_model(
                "Logistic Regression",
                y_test,
                logistic_preds,
                logistic_probs,
            ),
            evaluate_model(
                "Random Forest",
                y_test,
                rf_preds,
                rf_probs,
            ),
        ]
    )

    prediction_results = X_test.copy()
    prediction_results["actual_readmitted_30_days"] = y_test.values
    prediction_results["logistic_probability"] = logistic_probs
    prediction_results["logistic_prediction"] = logistic_preds
    prediction_results["random_forest_probability"] = rf_probs
    prediction_results["random_forest_prediction"] = rf_preds

    rf_classifier = rf_model.named_steps["classifier"]
    feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    logging.info("Logistic Regression Report:\n%s", classification_report(y_test, logistic_preds, zero_division=0))
    logging.info("Random Forest Report:\n%s", classification_report(y_test, rf_preds, zero_division=0))

    return model_results, prediction_results, feature_importance


def save_outputs(
    analysis_df: pd.DataFrame,
    model_results: pd.DataFrame,
    prediction_results: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    """Save analysis outputs to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df.to_csv(OUTPUT_DIR / "analysis_dataset.csv", index=False)
    model_results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    prediction_results.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    logging.info("Saved outputs to %s", OUTPUT_DIR)


def run_analysis() -> None:
    """Run the full analysis pipeline."""
    setup_logging()
    logging.info("Starting readmission analysis pipeline")

    collection = get_mongo_collection()

    raw_df = flatten_patient_documents(collection)
    analysis_df = engineer_features(raw_df)

    model_results, prediction_results, feature_importance = train_models(analysis_df)

    save_outputs(
        analysis_df,
        model_results,
        prediction_results,
        feature_importance,
    )

    logging.info("Analysis pipeline complete")
    print(model_results)


if __name__ == "__main__":
    run_analysis()