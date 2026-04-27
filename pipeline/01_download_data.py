"""
01_download_data.py

Downloads selected Synthea CSV files for the hospital readmission project
and saves them locally for later loading into MongoDB.
"""

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError


DATA_URL = "https://mitre.box.com/shared/static/aw9po06ypfb9hrau4jamtvtz0e5ziucz.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "logs"

ZIP_PATH = RAW_DATA_DIR / "synthea_csv.zip"

NEEDED_FILES = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "medications.csv",
    "procedures.csv",
    "observations.csv",
    "organizations.csv",
    "providers.csv",
]


def setup_logging() -> None:
    """Configure logging to both a log file and the console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_DIR / "download_data.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)


def download_zip() -> None:
    """Download the Synthea ZIP file if it does not already exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        logging.info("ZIP file already exists: %s", ZIP_PATH)
        return

    try:
        logging.info("Downloading data from %s", DATA_URL)
        urlretrieve(DATA_URL, ZIP_PATH)
        logging.info("Download complete: %s", ZIP_PATH)

    except HTTPError as e:
        logging.error("HTTP error while downloading data: %s", e)
        raise

    except URLError as e:
        logging.error("URL error while downloading data: %s", e)
        raise

    except Exception as e:
        logging.error("Unexpected error while downloading data: %s", e)
        raise


def extract_needed_csvs() -> None:
    """Extract only the CSV files needed for this project."""
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_contents = zip_ref.namelist()

            for filename in NEEDED_FILES:
                matches = [path for path in zip_contents if path.endswith(filename)]

                if not matches:
                    logging.warning("Could not find %s in ZIP file", filename)
                    continue

                source_path = matches[0]
                output_path = RAW_DATA_DIR / filename

                if output_path.exists():
                    logging.info("%s already extracted", filename)
                    continue

                with zip_ref.open(source_path) as source_file:
                    output_path.write_bytes(source_file.read())

                logging.info("Extracted %s to %s", filename, output_path)

    except zipfile.BadZipFile:
        logging.error("Downloaded file is not a valid ZIP file.")
        raise

    except Exception as e:
        logging.error("Unexpected error while extracting CSV files: %s", e)
        raise


def run_download() -> None:
    """Run the full raw data download and extraction process."""
    setup_logging()
    logging.info("Starting raw data acquisition process")

    download_zip()
    extract_needed_csvs()

    logging.info("Raw data acquisition process complete")


if __name__ == "__main__":
    run_download()