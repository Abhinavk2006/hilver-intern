import urllib.request
import zipfile
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

TWCS_CSV_PATH = RAW_DATA_DIR / "twcs.csv"
ZIP_PATH = RAW_DATA_DIR / "twcs.csv.zip"

def download_file():
    if TWCS_CSV_PATH.exists() and TWCS_CSV_PATH.stat().st_size > 1000000:
        print(f"Dataset already exists: {TWCS_CSV_PATH} ({TWCS_CSV_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return

    url = "https://huggingface.co/datasets/thoughtvector/customer_support_on_twitter/resolve/main/twcs.csv"
    print(f"Downloading {url} ...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
        else:
            sys.stdout.write(f"\rDownloaded {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(TWCS_CSV_PATH), reporthook=report_progress)
        print("\nDownload finished successfully!")
    except Exception as e:
        print(f"\nFailed downloading direct CSV: {e}")
        # Try downloading zip
        zip_url = "https://huggingface.co/datasets/thoughtvector/customer_support_on_twitter/resolve/main/twcs.csv.zip"
        print(f"Trying zip download: {zip_url}")
        urllib.request.urlretrieve(zip_url, str(ZIP_PATH), reporthook=report_progress)
        print("\nZip download finished. Extracting...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Extraction complete!")

if __name__ == "__main__":
    download_file()
