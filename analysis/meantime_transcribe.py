#!/usr/bin/env python3
"""
Transcribe a podcast's episodes with OpenAI Whisper, skipping episodes already transcribed.

Repaired copy. Changes from the published version:
- Runs locally as well as on the original GCP VM. With DISSERTATION_DATA_DIR set (see storage_backend.py), audio is
  read from <data>/audio/<podcast>/ and transcripts are written under DISSERTATION_OUTPUT_DIR, never into the data
  folder. Without it, it reads from and writes to the GCS bucket as before.
- The VM-only paths (/home/kingsdissert, /mnt/disks/data/logs) are now options.
- The Healthchecks ping URL and ntfy topic were hard-coded (and public in the repo). They are now optional
  environment variables, HEALTHCHECKS_URL and NTFY_TOPIC; without them no alerts are sent.
- Fixed a crash: if a podcast had no audio, the final cleanup referenced the stall timer before it existed.

Usage:
    python analysis/meantime_transcribe.py "<PODCAST_NAME>" [--model base] [--work-dir DIR] [--log-dir DIR]
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage_backend import destination, is_local, storage  # noqa: E402

GCS_BUCKET = "podcast-dissertation-audio"

parser = argparse.ArgumentParser(description="Transcribe one podcast's episodes with Whisper")
parser.add_argument("podcast_name")
parser.add_argument("--model", default="base", help="Whisper model size (default: base, as in the dissertation)")
parser.add_argument("--work-dir", default=None, help="Scratch folder for audio being transcribed (default: ./work/<podcast>)")
parser.add_argument("--log-dir", default="logs", help="Folder for the log file (default: ./logs)")
args = parser.parse_args()

PODCAST_NAME = args.podcast_name
LOCAL_AUDIO_DIR = Path(args.work_dir or f"work/{PODCAST_NAME}")
GCS_AUDIO_PREFIX = f"audio/{PODCAST_NAME}/"
GCS_TRANSCRIPT_PREFIX = f"podcast_transcripts/{PODCAST_NAME}/"
LOG_DIR = Path(args.log_dir)
MODEL_SIZE = args.model
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}" if NTFY_TOPIC else None
HEALTHCHECKS_URL = os.environ.get("HEALTHCHECKS_URL")

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"{PODCAST_NAME}_transcription.log"


def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")


def notify(title, message):
    if not NTFY_URL:
        return
    try:
        import requests
        response = requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": title.encode("utf-8"), "Content-Type": "text/plain; charset=utf-8"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        log(f"❌ Failed to send ntfy alert: {e}")


def ping():
    if not HEALTHCHECKS_URL:
        return
    try:
        import requests
        requests.get(HEALTHCHECKS_URL, timeout=10)
    except Exception as e:
        log(f"⚠️ Healthchecks ping failed: {e}")


def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


def list_files(prefix, suffix):
    blobs = storage.Client().list_blobs(GCS_BUCKET, prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith(suffix)]


def already_transcribed():
    names = set(os.path.basename(f) for f in list_files(GCS_TRANSCRIPT_PREFIX, ".json"))
    if is_local():  # transcripts made by earlier local runs live in the output folder
        from storage_backend import output_dir
        out = output_dir() / GCS_TRANSCRIPT_PREFIX
        if out.is_dir():
            names |= {p.name for p in out.glob("*.json")}
    return names


def download_blob(blob_name, destination_path):
    storage.Client().bucket(GCS_BUCKET).blob(blob_name).download_to_filename(destination_path)
    log(f"✅ Fetched {blob_name}")


def upload_blob(source, destination_blob_name):
    storage.Client().bucket(GCS_BUCKET).blob(destination_blob_name).upload_from_filename(source)
    log(f"✅ Saved {destination(GCS_BUCKET, destination_blob_name)}")


def transcribe_file(model, file_path):
    return model.transcribe(str(file_path), word_timestamps=True)


def send_stall_alert():
    notify("⏱️ Transcription stalled", f"No progress for 30 minutes in {PODCAST_NAME}")
    log("⏱️ Stall alert triggered by watchdog timer")


start_time = time.time()
crash_flag = True
stall_timer = None

log(f"Starting transcription for podcast: {PODCAST_NAME}")
notify(f"🚀 Transcription started: {PODCAST_NAME}", f"Started transcribing {PODCAST_NAME}")

try:
    LOCAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ping()
    log("📡 Script started")

    mp3s = list_files(GCS_AUDIO_PREFIX, ".mp3")
    done = already_transcribed()

    if not mp3s:
        log(f"⚠️ No .mp3 files found under {GCS_AUDIO_PREFIX}")
        notify("⚠️ No episodes found", f"{PODCAST_NAME} had no .mp3 files.")
        crash_flag = False
        sys.exit()

    import torch
    import whisper

    model = whisper.load_model(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
    stall_timer = threading.Timer(1800, send_stall_alert)
    stall_timer.start()

    for blob_name in mp3s:
        original_filename = os.path.basename(blob_name)
        sanitized_filename = sanitize_filename(original_filename)
        local_audio_path = LOCAL_AUDIO_DIR / sanitized_filename
        local_json_path = LOCAL_AUDIO_DIR / sanitized_filename.replace(".mp3", ".json")
        json_path = f"{GCS_TRANSCRIPT_PREFIX}{original_filename.replace('.mp3', '.json')}"

        if original_filename.replace(".mp3", ".json") in done:
            log(f"⏭ Skipping {original_filename}, already transcribed.")
            continue

        try:
            download_blob(blob_name, local_audio_path)
            result = transcribe_file(model, local_audio_path)
            with open(local_json_path, "w") as f:
                json.dump(result, f, indent=2)
            upload_blob(local_json_path, json_path)
            os.remove(local_audio_path)   # only the scratch copy; the original audio is untouched
            os.remove(local_json_path)
            log(f"✅ Done with {original_filename}")
            stall_timer.cancel()
            stall_timer = threading.Timer(1800, send_stall_alert)
            stall_timer.start()
            ping()
        except Exception as e:
            log(f"❌ Failed to process {original_filename}: {e}")
            notify("❌ Transcription Error", f"Failed to transcribe {original_filename}")

    crash_flag = False

except SystemExit:
    raise
except Exception as e:
    log(f"🔥 Script failed: {e}")
    notify("🔥 Script Error", str(e))
    raise

finally:
    duration = time.time() - start_time
    if stall_timer is not None:
        stall_timer.cancel()
        log("🛑 Stall timer cancelled — script complete")
    log(f"🏁 Done in {duration:.2f} seconds")
    ping()
    if crash_flag:
        notify("❌ Crash or VM interruption", f"Transcription for {PODCAST_NAME} failed or was cut off unexpectedly.")
    else:
        notify(f"🏁 Finished: {PODCAST_NAME}", f"All done in {int(duration)} seconds")
