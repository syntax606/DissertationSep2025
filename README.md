Podcast Influence Analysis

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/syntax606/DissertationSep2025)](https://github.com/syntax606/DissertationSep2025)

This repository contains the full computational framework and scripts for analyzing **influence operations in podcast media**.  
The methodology integrates:

- **Automatic transcription** (Whisper)
- **Enrichment** with metadata & entity tagging  
- **Persuasion & blame detection**  
- **Statistical analysis** with bootstrapping  

All scripts are modular and reproducible for research and dissertation work.



Repository Structure

```
podcast-influence-analysis/
├── analysis/   # Core analysis scripts
│   ├── meantime_transcribe.py
│   ├── enrichment_pipeline.py
│   ├── us_podcast_analysis_enhanced_timeline.py
│   ├── tenet_enhanced_analysis.py
│   ├── bootstrap_confidence_intervals.py
│   ├── recalculate_influence_scores.py
│   ├── equal_weighting_justification_analysis.py
│   └── political_discourse_analyzer2.py
│
├── data/       # Curated outputs used in dissertation & validation
│   ├── samples/        # Example outputs
│   │   ├── sample_output.json
│   │   └── sample_tenet.json
│   │
│   ├── stats/          # Aggregated statistical outputs
│   │   ├── master_stats_orientation_summary.csv
│   │   ├── master_stats_per_show.csv
│   │   ├── dissertation_stats_subset.json
│   │   └── figure_orientation_ci.png
│   │
│   ├── dissertation/   # Final curated results
│   │   ├── dissertation_key_findings.md
│   │   ├── political_figures_analysis.json
│   │   ├── russia_analysis_results.json
│   │   ├── framing_analysis_visualization.png
│   │   └── causal_analysis_blame_separated.png
│   │
│   └── testing/        # Validation & methodology tests
│       ├── bootstrap_methodology_transparent.txt
│       ├── computational_performance_formatted.txt
│       ├── per_show_influence_scores.csv
│       ├── weighting_justification.json
│       └── weighting_analysis.png
│
├── docs/       # Technical methodology and notes
│   └── Methodology_Master_Synopsis.md
│
└── README.md   # This file
```



Data Availability

This repository contains curated outputs from the full dissertation pipeline.  
The raw podcast transcripts and bulk per-episode JSONs are stored in Google Cloud Storage (GCS) due to size, but representative and summary files are included here.


`data/` Folder Overview

- **samples**  
  Small example outputs for transparency.  
  - `sample_output.json` – representative enhanced transcript analysis  
  - `sample_tenet.json` – Tenet baseline example

- **stats**  
  Aggregated statistical outputs.  
  - `master_stats_orientation_summary.csv` – normalized comparison across orientations  
  - `master_stats_per_show.csv` – per-show influence intensity scores  
  - `dissertation_stats_subset.json` – subset used in dissertation tables  
  - `figure_orientation_ci.png` – CI visualization

- **dissertation**  
  Final curated results used in the dissertation.  
  - `dissertation_key_findings.md` – high-level summary  
  - `political_figures_analysis.json` – entity-level blame & influence  
  - `russia_analysis_results.json` – Russia-specific analysis  
  - `framing_analysis_visualization.png` – persuasive framing visualization  
  - `causal_analysis_blame_separated.png` – causal blame separation

- **testing**  
  Methodology validation and bootstrap runs.  
  - `bootstrap_methodology_transparent.txt` – full bootstrap CI documentation  
  - `computational_performance_formatted.txt` – runtime performance validation  
  - `per_show_influence_scores.csv` – episode-level influence scores  
  - `weighting_justification.json` / `weighting_analysis.png` – equal-weighting validation



Larger Data

- Full transcripts (`enriched_transcripts/`) and batch enhanced analyses (`enhanced_analysis/`) are stored in the **private GCS bucket**:  
  `gs://podcast-dissertation-audio/`  
- Contact the author if access is required for replication.




Key Scripts

- **`meantime_transcribe.py`** → Transcribes podcasts using OpenAI Whisper.  
- **`enrichment_pipeline.py`** → Enriches transcripts with entity detection, metadata, and rhetorical markers.  
- **`bootstrap_confidence_intervals.py`** → Runs statistical bootstrapping for confidence intervals.  
- **`recalculate_influence_scores.py`** → Recomputes influence ratios with deduplication.  
- **`equal_weighting_justification_analysis.py`** → Evaluates different weighting schemes for influence metrics.  
- **`political_discourse_analyzer2.py`** → Full discourse analyzer: blame, persuasion, and influence metrics.  
- **`tenet_enhanced_analysis.py`** / **`us_podcast_analysis_enhanced_timeline.py`** → Orientation- and timeline-specific analyses.  




Documentation

See [`docs/Methodology_Master_Synopsis.md`](docs/Methodology_Master_Synopsis.md) for full methodological details, including:  
- Influence score formula  
- Persuasion and blame detection  
- Bootstrapping methodology  
- Temporal and orientation-specific analysis  




Setup & Usage

Clone this repository and install dependencies:

```bash
git clone https://github.com/syntax606/DissertationSep2025.git
cd DissertationSep2025
pip install -r requirements.txt

Run an analysis script, e.g.:
python analysis/bootstrap_confidence_intervals.py --bootstrap 1000
```

Due to size and copyright restrictions, full transcripts are hosted on Google Cloud Storage. Researchers may request access by contacting me.



License

This project is licensed under the MIT License. See the LICENSE file for details.
