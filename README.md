# Podcast Influence Analysis

This repository contains the computational framework and scripts used for analyzing influence operations in podcast media.  
The project combines transcription, enrichment, persuasion/blame detection, and statistical analysis into a reproducible pipeline.

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
- **meantime_transcribe.py** → Transcribes podcasts using OpenAI Whisper.
- **enrichment_pipeline.py** → Adds metadata, entity tagging, and persuasive language detection.
- **bootstrap_confidence_intervals.py** → Runs statistical bootstrapping for influence score validation.
- **recalculate_influence_scores.py** → Cleans and recalculates influence scores from enhanced files.
- **equal_weighting_justification_analysis.py** → Tests weighting schemes for influence metrics.
- **political_discourse_analyzer2.py** → Full discourse analysis including blame, persuasion, and influence.
- **tenet_enhanced_analysis.py** / **us_podcast_analysis_enhanced_timeline.py** → Orientation-specific analyses.

---

## 📖 Documentation
See the `docs/` folder for full methodology notes and explanations.

