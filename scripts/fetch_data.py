import os
import sys
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TWCS_CSV_PATH, RAW_DATA_DIR

def download_twcs_dataset():
    if TWCS_CSV_PATH.exists():
        print(f"Dataset already exists at {TWCS_CSV_PATH} ({TWCS_CSV_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return TWCS_CSV_PATH

    print("Downloading Customer Support on Twitter dataset from HuggingFace Hub...")
    # Dataset repo: thoughtvector/customer_support_on_twitter
    try:
        csv_path = hf_hub_download(
            repo_id="thoughtvector/customer_support_on_twitter",
            filename="twcs.csv",
            repo_type="dataset",
            local_dir=str(RAW_DATA_DIR)
        )
        print(f"Downloaded twcs.csv to {csv_path}")
        return Path(csv_path)
    except Exception as e:
        print(f"Error downloading twcs.csv: {e}")
        # Try downloading zip if available
        try:
            zip_path = hf_hub_download(
                repo_id="thoughtvector/customer_support_on_twitter",
                filename="twcs.csv.zip",
                repo_type="dataset",
                local_dir=str(RAW_DATA_DIR)
            )
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            print(f"Extracted twcs.csv from zip to {TWCS_CSV_PATH}")
            return TWCS_CSV_PATH
        except Exception as e2:
            print(f"Failed zip fallback: {e2}")
            raise e

if __name__ == "__main__":
    download_twcs_dataset()
