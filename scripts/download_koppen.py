"""Download Beck et al. 2018 Köppen-Geiger 1km raster and upload to GCS."""
import os
import sys
import requests
from google.cloud import storage

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
LOCAL_PATH = os.path.join(OUT_DIR, "koppen_beck2018_1km.tif")
GCS_BUCKET = "dctii-raw-oil-tank-monitoring-123"
GCS_BLOB = "static/koppen_beck2018_1km.tif"


def get_download_url():
    """Resolve actual download URL from figshare API."""
    # Try direct file redirect
    r = requests.get(
        "https://api.figshare.com/v2/file/download/12407516",
        allow_redirects=False, timeout=30,
    )
    if r.status_code in (301, 302, 303, 307):
        return r.headers["Location"]

    # Fall back to article files endpoint
    r2 = requests.get(
        "https://api.figshare.com/v2/articles/6396959/files", timeout=30,
    )
    r2.raise_for_status()
    for f in r2.json():
        if "1km" in f["name"].lower() and f["name"].endswith(".tif"):
            return f["download_url"]
        if str(f["id"]) == "12407516":
            return f["download_url"]

    # Last resort: try ndownloader with retries (202 = queued)
    for attempt in range(5):
        import time
        r3 = requests.get(
            "https://figshare.com/ndownloader/files/12407516",
            allow_redirects=True, stream=True, timeout=120,
        )
        if r3.status_code == 200 and int(r3.headers.get("Content-Length", "0")) > 1000:
            return r3.url
        print(f"  Attempt {attempt+1}: status {r3.status_code}, waiting 10s...")
        time.sleep(10)

    raise RuntimeError("Could not resolve download URL for Köppen raster")


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    url = get_download_url()
    print(f"Downloading from: {url[:120]}...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = 0
    with open(LOCAL_PATH, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            f.write(chunk)
            total += len(chunk)
            print(f"  {total / 1024 / 1024:.1f} MB", end="\r")
    print(f"\nDownloaded {total / 1024 / 1024:.1f} MB -> {LOCAL_PATH}")
    return total


def upload_to_gcs():
    client = storage.Client(project="oil-tank-monitoring-123")
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_BLOB)
    print(f"Uploading to gs://{GCS_BUCKET}/{GCS_BLOB} ...")
    blob.upload_from_filename(LOCAL_PATH, timeout=600)
    print(f"Upload complete: gs://{GCS_BUCKET}/{GCS_BLOB}")


if __name__ == "__main__":
    size = download()
    if size < 1000:
        print("ERROR: Downloaded file is too small, likely not the raster.")
        sys.exit(1)
    upload_to_gcs()
    print("Done!")
