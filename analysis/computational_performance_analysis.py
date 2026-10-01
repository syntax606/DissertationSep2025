#!/usr/bin/env python3
"""
Measure how long the analysis pipeline actually takes, on real transcripts, on this machine.

Rewritten. The published version (computational_performance_analysis.py, whose output was reported as "0.54 seconds
per episode") did not run the pipeline: each stage was a time.sleep() call (for example 0.005 s x 50 segments =
0.25 s for "causal extraction"), and transcription was not included. Those figures were not measurements.

This version times the real code:
- scoring: the causal, keyword and persuasive analysis from tenet_enhanced_analysis.py, run on real enriched
  transcripts (always);
- enrichment models: sentence-BERT, the emotion and sentiment models from enrichment_pipeline.py, run on every
  segment of a sample of episodes (with --models; needs sentence-transformers and transformers);
- transcription: Whisper on real audio files, reported as a real-time factor (with --audio; needs openai-whisper).

It reports the hardware and software it ran on, since timings only mean something alongside them.

Usage:
    export DISSERTATION_DATA_DIR="/Volumes/Caro Drive/podcast-dissertation-audio"
    python analysis/computational_performance_analysis.py --episodes 100 [--models --model-episodes 5] \\
        [--audio "/Volumes/Caro Drive/podcast-dissertation-audio/audio/<show>/<file>.mp3" ...]
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage_backend import storage  # noqa: E402

BUCKET_NAME = "podcast-dissertation-audio"


def machine():
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["accelerator"] = ("cuda" if torch.cuda.is_available()
                               else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                               else "cpu")
    except ImportError:
        pass
    return info


def summarise(values):
    values = sorted(values)
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))], "total": sum(values)}


def time_scoring(files, bucket):
    import tenet_enhanced_analysis as tea  # the scoring functions used for the published scores

    rows = []
    for name in files:
        t0 = time.perf_counter()
        data = json.loads(bucket.blob(name).download_as_string())
        t1 = time.perf_counter()
        text = tea.extract_full_text(data.get("segments", []))
        words = len(text.split())
        causal, _ = tea.extract_causal_relationships(text)
        t2 = time.perf_counter()
        tea.analyze_keywords_detailed(text, tea.KEYWORD_TOPICS)
        t3 = time.perf_counter()
        tea.analyze_keywords_detailed(text, tea.PERSUASIVE_TECHNIQUES)
        t4 = time.perf_counter()
        rows.append({"file": name, "words": words, "segments": len(data.get("segments", [])),
                     "load_s": t1 - t0, "causal_s": t2 - t1, "topics_s": t3 - t2, "persuasive_s": t4 - t3,
                     "scoring_s": t4 - t1})
    return rows


def load_available_models():
    """The enrichment models from enrichment_pipeline.py, skipping any whose library isn't installed."""
    models, missing = {}, []
    try:
        from sentence_transformers import SentenceTransformer
        models["embedding"] = SentenceTransformer("all-mpnet-base-v2").encode
    except ImportError:
        missing.append("embedding (pip install sentence-transformers)")
    try:
        from transformers import pipeline
        models["emotion"] = pipeline("text-classification", "j-hartmann/emotion-english-distilroberta-base", top_k=None)
        models["sentiment"] = pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
    except ImportError:
        missing.append("emotion and sentiment (pip install transformers torch)")
    return models, missing


def time_models(files, bucket):
    t0 = time.perf_counter()
    models, missing = load_available_models()
    load_s = time.perf_counter() - t0
    rows = []
    for name in files:
        segments = [s for s in json.loads(bucket.blob(name).download_as_string()).get("segments", [])
                    if len(s.get("text", "").strip()) >= 10]
        timings = {f"{k}_s": 0.0 for k in models}
        for s in segments:  # one segment at a time, as enrichment_pipeline.py does
            for k, fn in models.items():
                a = time.perf_counter()
                fn(s["text"])
                timings[f"{k}_s"] += time.perf_counter() - a
        rows.append({"file": name, "segments": len(segments), **timings, "models_s": sum(timings.values())})
    return load_s, rows, list(models), missing


def time_whisper(paths, model_size):
    import whisper

    t0 = time.perf_counter()
    model = whisper.load_model(model_size)
    load_s = time.perf_counter() - t0
    rows = []
    for p in paths:
        audio_s = whisper.audio.load_audio(p).shape[0] / whisper.audio.SAMPLE_RATE
        a = time.perf_counter()
        model.transcribe(p, word_timestamps=True)
        took = time.perf_counter() - a
        rows.append({"file": p, "audio_s": audio_s, "transcribe_s": took, "real_time_factor": took / audio_s})
    return load_s, rows


def main():
    parser = argparse.ArgumentParser(description="Measure the pipeline's real processing time")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to score (random sample; default 100)")
    parser.add_argument("--models", action="store_true", help="Also time the enrichment models")
    parser.add_argument("--model-episodes", type=int, default=3, help="Episodes to run the models on (default 3)")
    parser.add_argument("--audio", nargs="*", default=[], help="Audio files to time Whisper on")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/performance")
    args = parser.parse_args()

    bucket = storage.Client().bucket(BUCKET_NAME)
    all_files = [b.name for b in bucket.list_blobs(prefix="enriched_transcripts/") if b.name.endswith("_enriched.json")]
    if not all_files:
        sys.exit("No enriched transcripts found under enriched_transcripts/")
    rng = random.Random(args.seed)
    sample = rng.sample(all_files, min(args.episodes, len(all_files)))

    result = {"generated": datetime.now().isoformat(timespec="seconds"), "machine": machine(),
              "corpus_files_found": len(all_files)}

    print(f"Timing scoring on {len(sample)} of {len(all_files)} episodes…")
    rows = time_scoring(sample, bucket)
    words = [r["words"] for r in rows]
    result["scoring"] = {
        "episodes": len(rows), "mean_words": statistics.fmean(words),
        "per_episode_s": summarise([r["scoring_s"] for r in rows]),
        "load_per_episode_s": summarise([r["load_s"] for r in rows]),
        "per_1k_words_s": sum(r["scoring_s"] for r in rows) / (sum(words) / 1000),
        "stages_mean_s": {k: statistics.fmean(r[k] for r in rows) for k in ("causal_s", "topics_s", "persuasive_s")},
    }

    if args.models:
        print(f"Timing enrichment models on {args.model_episodes} episodes…")
        load_s, mrows, timed, missing = time_models(sample[:args.model_episodes], bucket)
        segs = sum(r["segments"] for r in mrows)
        result["models"] = {
            "timed": timed, "not_timed": missing,
            "load_s": load_s, "episodes": len(mrows), "segments": segs,
            "per_episode_s": summarise([r["models_s"] for r in mrows]),
            "per_segment_s": sum(r["models_s"] for r in mrows) / segs if segs else None,
            "by_model_per_segment_s": {k: sum(r[f"{k}_s"] for r in mrows) / segs for k in timed} if segs else None,
        }

    if args.audio:
        print(f"Timing Whisper ({args.whisper_model}) on {len(args.audio)} audio files…")
        load_s, arows = time_whisper(args.audio, args.whisper_model)
        result["transcription"] = {"model": args.whisper_model, "load_s": load_s, "files": arows,
                                   "mean_real_time_factor": statistics.fmean(r["real_time_factor"] for r in arows)}

    s = result["scoring"]
    lines = [
        "PIPELINE TIMING (MEASURED)",
        "=" * 40,
        f"Machine: {result['machine']['platform']}, {result['machine']['cpu_count']} CPUs, Python {result['machine']['python']}"
        + (f", {result['machine'].get('accelerator')}" if result['machine'].get('accelerator') else ""),
        "",
        f"Scoring (causal + keyword + persuasive), {s['episodes']} episodes averaging {s['mean_words']:,.0f} words:",
        f"  median {s['per_episode_s']['median']:.3f} s per episode (mean {s['per_episode_s']['mean']:.3f}, 90th pct {s['per_episode_s']['p90']:.3f})",
        f"  {s['per_1k_words_s'] * 1000:.2f} ms per 1,000 words; reading the file adds a median {s['load_per_episode_s']['median']:.3f} s",
    ]
    if "models" in result:
        m = result["models"]
        lines += ["", f"Enrichment models ({', '.join(m['timed'])}) on {m['episodes']} episodes ({m['segments']:,} segments), "
                  f"after {m['load_s']:.1f} s loading:",
                  f"  {m['per_segment_s']:.3f} s per segment; median {m['per_episode_s']['median']:.1f} s per episode"]
        lines += [f"  by model: " + ", ".join(f"{k} {v * 1000:.1f} ms" for k, v in m["by_model_per_segment_s"].items()) + " per segment"]
        if m["not_timed"]:
            lines += ["  not timed (library missing): " + "; ".join(m["not_timed"])]
    else:
        lines += ["", "Enrichment models: not timed (run with --models)."]
    if "transcription" in result:
        t = result["transcription"]
        lines += ["", f"Whisper {t['model']}: real-time factor {t['mean_real_time_factor']:.2f} "
                  f"(1.0 = as long as the audio) over {len(t['files'])} files"]
    else:
        lines += ["", "Transcription: not timed (run with --audio)."]
    text = "\n".join(lines)
    print("\n" + text)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"performance_{stamp}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, f"performance_{stamp}.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
