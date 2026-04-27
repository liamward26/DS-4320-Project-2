"""
02_load_mongo.py

Loads selected Synthea CSV files into MongoDB using a patient-centered
document model for hospital readmission analysis.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"

DB_NAME = "hospital_readmissions"
COLLECTION_NAME = "patients"

CSV_FILES = {
    "patients": "patients.csv",
    "encounters": "encounters.csv",
    "conditions": "conditions.csv",
    "medications": "medications.csv",
    "procedures": "procedures.csv",
    "observations": "observations.csv",
    "organizations": "organizations.csv",
    "providers": "providers.csv",
}


def setup_logging() -> None:
    """Configure logging to both a file and the console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_DIR / "load_mongo.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def read_csv(filename: str) -> pd.DataFrame:
    """Read a CSV file from the raw data directory."""
    path = RAW_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    logging.info("Reading %s", path)
    return pd.read_csv(path, low_memory=False)


def clean_nan(value):
    """Convert NaN values into None so MongoDB stores valid nulls."""
    if pd.isna(value):
        return None
    return value


def row_to_dict(row: pd.Series) -> dict:
    """Convert a pandas row to a Mongo-safe dictionary."""
    return {key: clean_nan(value) for key, value in row.items()}


def calculate_readmissions(encounters: pd.DataFrame) -> pd.DataFrame:
    """
    Add a readmitted_30_days flag to inpatient encounters.

    A readmission is counted when the same patient has another encounter
    starting within 30 days after a prior encounter ends.
    """
    encounters = encounters.copy()

    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce")
    encounters["STOP"] = pd.to_datetime(encounters["STOP"], errors="coerce")
    encounters["readmitted_30_days"] = False

    encounters = encounters.sort_values(["PATIENT", "START"])

    for patient_id, group in encounters.groupby("PATIENT"):
        group = group.sort_values("START")

        for idx, encounter in group.iterrows():
            stop_time = encounter["STOP"]

            if pd.isna(stop_time):
                continue

            future_visits = group[group["START"] > stop_time]

            if future_visits.empty:
                continue

            next_start = future_visits.iloc[0]["START"]
            days_until_next = (next_start - stop_time).days

            if 0 <= days_until_next <= 30:
                encounters.loc[idx, "readmitted_30_days"] = True

    return encounters


def build_patient_documents(data: dict[str, pd.DataFrame]) -> list[dict]:
    """Build one MongoDB document per patient with nested clinical history."""
    patients = data["patients"]
    encounters = calculate_readmissions(data["encounters"])

    conditions = data["conditions"]
    medications = data["medications"]
    procedures = data["procedures"]
    observations = data["observations"]
    organizations = data["organizations"]
    providers = data["providers"]

    patient_documents = []

    logging.info("Building patient-centered documents")

    for _, patient in patients.iterrows():
        patient_id = patient["Id"]

        patient_encounters = encounters[encounters["PATIENT"] == patient_id]

        encounter_docs = []

        for _, encounter in patient_encounters.iterrows():
            encounter_id = encounter["Id"]

            encounter_doc = row_to_dict(encounter)

            encounter_doc["START"] = (
                encounter["START"].isoformat() if pd.notna(encounter["START"]) else None
            )
            encounter_doc["STOP"] = (
                encounter["STOP"].isoformat() if pd.notna(encounter["STOP"]) else None
            )

            encounter_doc["conditions"] = [
                row_to_dict(row)
                for _, row in conditions[
                    conditions["ENCOUNTER"] == encounter_id
                ].iterrows()
            ]

            encounter_doc["medications"] = [
                row_to_dict(row)
                for _, row in medications[
                    medications["ENCOUNTER"] == encounter_id
                ].iterrows()
            ]

            encounter_doc["procedures"] = [
                row_to_dict(row)
                for _, row in procedures[
                    procedures["ENCOUNTER"] == encounter_id
                ].iterrows()
            ]

            encounter_doc["observations"] = [
                row_to_dict(row)
                for _, row in observations[
                    observations["ENCOUNTER"] == encounter_id
                ].iterrows()
            ]

            encounter_docs.append(encounter_doc)

        patient_doc = row_to_dict(patient)
        patient_doc["_id"] = patient_id
        patient_doc["encounters"] = encounter_docs

        patient_documents.append(patient_doc)

    metadata_doc = {
        "_id": "database_metadata",
        "source": "Synthea synthetic patient CSV data",
        "document_model": "One document per patient with embedded encounters and clinical records",
        "included_csvs": list(CSV_FILES.values()),
        "collections": {
            COLLECTION_NAME: "Patient-centered documents for readmission analysis"
        },
    }

    patient_documents.append(metadata_doc)

    logging.info("Built %s documents", len(patient_documents))
    return patient_documents


def load_to_mongo(documents: list[dict]) -> None:
    """Connect to MongoDB and insert patient documents."""
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        raise ValueError("MONGO_URI not found. Add it to your .env file.")

    try:
        client = MongoClient(mongo_uri)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        logging.info("Connected to MongoDB database: %s", DB_NAME)

        collection.delete_many({})
        logging.info("Cleared existing documents from %s", COLLECTION_NAME)

        if documents:
            collection.insert_many(documents)
            logging.info("Inserted %s documents into MongoDB", len(documents))

        collection.create_index("Id")
        collection.create_index("encounters.Id")
        collection.create_index("encounters.readmitted_30_days")

        logging.info("Created indexes")

        client.close()

    except PyMongoError as e:
        logging.error("MongoDB error: %s", e)
        raise

    except Exception as e:
        logging.error("Unexpected error while loading MongoDB: %s", e)
        raise


def load_mongo() -> None:
    """Run the full MongoDB loading process."""
    setup_logging()
    logging.info("Starting MongoDB loading process")

    data = {
        table_name: read_csv(filename)
        for table_name, filename in CSV_FILES.items()
    }

    documents = build_patient_documents(data)
    load_to_mongo(documents)

    logging.info("MongoDB loading process complete")


if __name__ == "__main__":
    load_mongo()