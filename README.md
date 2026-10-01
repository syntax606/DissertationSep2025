# Podcast Influence Analysis

Code and curated outputs for *Rhetorical Fingerprinting* (C. Swartz, MA dissertation, King's College London, 2025):
a six-dimensional analysis of persuasive structure in 6,437 US political podcast episodes, including Tenet Media's
catalogue.

**See [`CORRECTIONS.md`](CORRECTIONS.md)** for what was repaired in this version, the measured replacements for the
weighting and timing analyses, and the claims that could not be checked.

The pipeline covers:
- transcription with Whisper;
- enrichment of every ~6-second segment: embeddings, emotion, sentiment, persuasive tags and topics;
- influence intensity, (causal + persuasive) / words × 1,000, with keyword and blame analysis;
- bootstrap confidence intervals.

## Repository structure

```
analysis/
  storage_backend.py                      run on a local copy of the data, or on the original GCS bucket
  meantime_transcribe.py                  Whisper transcription
  enrichment_pipeline.py                  segment enrichment (embeddings, emotion, sentiment, persuasion, topics)
  tenet_enhanced_analysis.py              influence scores and keyword analysis, Tenet Media
  us_podcast_analysis_enhanced_timeline.py   the same for a US show
  transparent_bootstrap.py                the published confidence intervals (Chapter 4)
  per_show_influence_bootstrap.py         per-show scores (Table 4.1)
  recalculate_influence_scores.py         orientation and show summaries with bootstrap
  bootstrap_confidence_intervals.py       a simplified variant of the score (not the published metric)
  equal_weighting_justification_analysis.py  weighting schemes compared on the real data
  computational_performance_analysis.py   measured processing time
  political_discourse_analyzer2.py        entity-level blame and discourse analysis
validation/                               human validation kit: blind sample, annotation workbooks, scoring
data/
  reproduction/   results re-measured in September 2026 (bootstrap, weighting, timing)
  samples/  stats/  dissertation/  testing/   curated outputs (see CORRECTIONS.md for which to trust)
docs/
  Methodology_Master_Synopsis.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the analysis scripts. Transcription and enrichment need extra packages, listed there as
comments.

## Running on the local data

The data is laid out as in the original bucket (`enriched_transcripts/`, `tenet_enhanced_analysis/`,
`us_podcast_enhanced_analysis/`, `audio/`…). Point the scripts at a local copy:

```bash
export DISSERTATION_DATA_DIR="/path/to/podcast-dissertation-audio"
export DISSERTATION_OUTPUT_DIR="$PWD/outputs"      # optional; anything the scripts "upload" goes here

python analysis/transparent_bootstrap.py                                   # Chapter 4 intervals
python analysis/per_show_influence_bootstrap.py                            # Table 4.1
python analysis/equal_weighting_justification_analysis.py                  # weighting schemes, real data
python analysis/computational_performance_analysis.py --episodes 100 --models   # timing
python analysis/tenet_enhanced_analysis.py                                 # re-score Tenet from transcripts
python analysis/us_podcast_analysis_enhanced_timeline.py --show glenbeck --orientation "Right wing"
python analysis/enrichment_pipeline.py podcast_transcripts/<show>/ -o enriched_transcripts_new/
python analysis/meantime_transcribe.py "<PODCAST_NAME>"
```

To redo the human validation, see [`validation/README.md`](validation/README.md).

The scripts never write into the data folder. Without `DISSERTATION_DATA_DIR` they read from and write to
`gs://podcast-dissertation-audio` (needs `google-cloud-storage` and access to the bucket).

## Data availability

The full transcripts and per-episode files are not in this repository, because of their size and copyright. The
`data/` folder holds summaries and samples. Contact the author about access for replication.

## Licence

MIT: see [LICENSE](LICENSE).
