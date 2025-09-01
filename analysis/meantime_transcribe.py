import os
import sys
import time
import json
import whisper
import torch
import requests
import re
import threading
from pathlib import Path
from google.cloud import storage

# ✅ Constants
if len(sys.argv) < 2:
    print("Usage: python meantime_transcribe.py '<PODCAST_NAME>'")
    sys.exit(1)

PODCAST_NAME = sys.argv[1]
LOCAL_AUDIO_DIR = Path(f"/home/kingsdissert/audio/{PODCAST_NAME}")
GCS_BUCKET = "podcast-dissertation-audio"
GCS_AUDIO_PREFIX = f"audio/{PODCAST_NAME}/"
GCS_TRANSCRIPT_PREFIX = f"podcast_transcripts/{PODCAST_NAME}/"
LOG_DIR = Path("/mnt/disks/data/logs")
MODEL_SIZE = "base"
NTFY_TOPIC = "transcription-alerts"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
HEALTHCHECKS_URL = "https://hc-ping.com/dcc8edf4-aaf3-4ef3-950e-9f636f3023ee"

# ✅ Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"{PODCAST_NAME}_transcription.log"

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def notify(title, message):
    try:
        response = requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={
                "Title": title.encode("utf-8"),
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        response.raise_for_status()
    except Exception as e:
        log(f"❌ Failed to send ntfy alert: {e}")

def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)

def list_gcs_files(bucket_name, prefix, suffix):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith(suffix)]

def download_blob(bucket_name, blob_name, destination):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination)
    log(f"✅ Downloaded {blob_name}")

def upload_blob(bucket_name, source, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source)
    log(f"✅ Uploaded {destination_blob_name}")

def transcribe_file(model, file_path):
    return model.transcribe(str(file_path), word_timestamps=True)

def send_stall_alert():
    notify("⏱️ Transcription stalled", f"No progress for 30 minutes in {PODCAST_NAME}")
    log("⏱️ Stall alert triggered by watchdog timer")

# ✅ Start
start_time = time.time()
crash_flag = True

log(f"Starting transcription for podcast: {PODCAST_NAME}")
notify(f"🚀 Transcription started: {PODCAST_NAME}", f"Started transcribing {PODCAST_NAME}")

try:
    LOCAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    requests.get(HEALTHCHECKS_URL)
    log("📡 Healthchecks ping: script started")

    gcs_mp3s = list_gcs_files(GCS_BUCKET, GCS_AUDIO_PREFIX, ".mp3")
    gcs_jsons = set(os.path.basename(f) for f in list_gcs_files(GCS_BUCKET, GCS_TRANSCRIPT_PREFIX, ".json"))

    if not gcs_mp3s:
        log("⚠️ No .mp3 files found in GCS bucket.")
        notify("⚠️ No episodes found", f"{PODCAST_NAME} had no .mp3 files in GCS.")
        crash_flag = False
        sys.exit()

    model = whisper.load_model(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")
    stall_timer = threading.Timer(1800, send_stall_alert)
    stall_timer.start()

    for blob_name in gcs_mp3s:
        if time.time() - start_time > 10:
            requests.get(HEALTHCHECKS_URL)

        original_filename = os.path.basename(blob_name)
        sanitized_filename = sanitize_filename(original_filename)
        local_audio_path = LOCAL_AUDIO_DIR / sanitized_filename
        local_json_path = LOCAL_AUDIO_DIR / sanitized_filename.replace(".mp3", ".json")
        gcs_json_path = f"{GCS_TRANSCRIPT_PREFIX}{original_filename.replace('.mp3', '.json')}"

        if original_filename.replace(".mp3", ".json") in gcs_jsons:
            log(f"⏭ Skipping {original_filename}, already transcribed in GCS.")
            continue

        try:
            download_blob(GCS_BUCKET, blob_name, local_audio_path)
            result = transcribe_file(model, local_audio_path)
            with open(local_json_path, "w") as f:
                json.dump(result, f, indent=2)
            upload_blob(GCS_BUCKET, local_json_path, gcs_json_path)
            os.remove(local_audio_path)
            os.remove(local_json_path)
            log(f"✅ Done with {original_filename}")
            stall_timer.cancel()
            stall_timer = threading.Timer(1800, send_stall_alert)
            stall_timer.start()
            requests.get(HEALTHCHECKS_URL)

        except Exception as e:
            log(f"❌ Failed to process {original_filename}: {e}")
            notify("❌ Transcription Error", f"Failed to transcribe {original_filename}")

    crash_flag = False

except Exception as e:
    log(f"🔥 Script failed: {e}")
    notify("🔥 Script Error", str(e))
    raise

finally:
    duration = time.time() - start_time
    stall_timer.cancel()
    log("🛑 Stall timer cancelled — script complete")
    log(f"🏁 Done in {duration:.2f} seconds")
    requests.get(HEALTHCHECKS_URL)
    if crash_flag:
        notify("❌ Crash or VM interruption", f"Transcription for {PODCAST_NAME} failed or was cut off unexpectedly.")
    else:
        notify(f"🏁 Finished: {PODCAST_NAME}", f"All done in {int(duration)} seconds")
