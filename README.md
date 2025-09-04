# 🎙️ Podcast Influence Analysis

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/syntax606/DissertationSep2025)](https://github.com/syntax606/DissertationSep2025)

This repository contains the full computational framework and scripts for analyzing **influence operations in podcast media**.  
The methodology integrates:

- 🎧 **Automatic transcription** (Whisper)
- 🏷 **Enrichment** with metadata & entity tagging  
- 🗣 **Persuasion & blame detection**  
- 📊 **Statistical analysis** with bootstrapping  

All scripts are modular and reproducible for research and dissertation work.

---

## 📂 Repository Structure
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
├── docs/       # Technical methodology and notes
│   └── Methodology_Master_Synopsis.md
│
└── README.md   # This file
---

## 🚀 Key Scripts

- **`meantime_transcribe.py`** → Transcribes podcasts using OpenAI Whisper.  
- **`enrichment_pipeline.py`** → Enriches transcripts with entity detection, metadata, and rhetorical markers.  
- **`bootstrap_confidence_intervals.py`** → Runs statistical bootstrapping for confidence intervals.  
- **`recalculate_influence_scores.py`** → Recomputes influence ratios with deduplication.  
- **`equal_weighting_justification_analysis.py`** → Evaluates different weighting schemes for influence metrics.  
- **`political_discourse_analyzer2.py`** → Full discourse analyzer: blame, persuasion, and influence metrics.  
- **`tenet_enhanced_analysis.py`** / **`us_podcast_analysis_enhanced_timeline.py`** → Orientation- and timeline-specific analyses.  

---

## 📖 Documentation

See [`docs/Methodology_Master_Synopsis.md`](docs/Methodology_Master_Synopsis.md) for full methodological details, including:  
- Influence score formula  
- Persuasion and blame detection  
- Bootstrapping methodology  
- Temporal and orientation-specific analysis  

---

## 🔧 Setup & Usage

Clone this repository and install dependencies:

```bash
git clone https://github.com/syntax606/DissertationSep2025.git
cd DissertationSep2025
pip install -r requirements.txt

Run an analysis script, e.g.:
python analysis/bootstrap_confidence_intervals.py --bootstrap 1000

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
