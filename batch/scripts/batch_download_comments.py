import csv
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "default"
CSV_FILE = Path(os.getenv("VODS_CSV", str(DEFAULT_DATA_DIR / "batch_twitch_vods_all.csv")))
COMMENTS_DIR = Path(os.getenv("COMMENTS_DIR", str(DEFAULT_DATA_DIR / "comments")))

downloader_env = os.getenv("TWITCH_DOWNLOADER_CLI")
if downloader_env:
    TWITCH_DOWNLOADER = Path(downloader_env).expanduser()
    if not TWITCH_DOWNLOADER.is_absolute():
        TWITCH_DOWNLOADER = PROJECT_ROOT / TWITCH_DOWNLOADER
else:
    TWITCH_DOWNLOADER = PROJECT_ROOT / "library" / "TwitchDownloaderCLI"

if not TWITCH_DOWNLOADER.exists():
    raise FileNotFoundError(
        f"TwitchDownloaderCLI not found at {TWITCH_DOWNLOADER}. "
        "Set TWITCH_DOWNLOADER_CLI to override."
    )

if not os.access(TWITCH_DOWNLOADER, os.X_OK):
    raise PermissionError(f"TwitchDownloaderCLI is not executable: {TWITCH_DOWNLOADER}")

COMMENTS_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"VOD CSV not found: {CSV_FILE}. "
        "Run get_vod_list_batch.py first or set VODS_CSV to override."
    )

with CSV_FILE.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row["url"]
        vod_id = url.split("/")[-1]
        output_file = COMMENTS_DIR / f"{vod_id}.json"
        if output_file.exists():
            print(f"Skipping {vod_id}, already exists")
            continue

        command = [str(TWITCH_DOWNLOADER), "chatdownload", "--id", url, "-o", str(output_file)]
        try:
            subprocess.run(command, check=True)
            print(f"Downloaded comments for {vod_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {vod_id}: {e}")
