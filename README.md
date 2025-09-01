# Podcast Influence Analysis

This repository contains the full computational framework and scripts used for **real-time detection of foreign influence operations in podcast media**.  
The methodology combines audio transcription, enrichment, persuasion/blame detection, and statistical analysis into a reproducible pipeline.

---

## 📂 Repository Structure
podcast-influence-analysis/
│
├── analysis/                  # Core analysis scripts
│   ├── meantime_transcribe.py
│   ├── enrichment_pipeline.py
│   ├── us_podcast_analysis_enhanced_timeline.py
│   ├── tenet_enhanced_analysis.py
│   ├── bootstrap_confidence_intervals.py
│   ├── recalculate_influence_scores.py
│   ├── equal_weighting_justification_analysis.py
│   └── political_discourse_analyzer2.py
│
├── docs/
│   └── Methodology_Master_Synopsis.md  # Full technical methodology
│
└── README.md                # This file
---

## 🔑 Key Scripts

- **meantime_transcribe.py**  
  Runs OpenAI Whisper transcription on podcast episodes.

- **enrichment_pipeline.py**  
  Embedding, sentiment, emotion, and persuasion enrichment.

- **us_podcast_analysis_enhanced_timeline.py**  
  Temporal evolution analysis of U.S. podcasts.

- **tenet_enhanced_analysis.py**  
  Baseline analysis of foreign "Tenet" podcasts.

- **bootstrap_confidence_intervals.py**  
  Statistical validation of influence scores (bootstrap resampling).

- **recalculate_influence_scores.py**  
  Recalculates episode-level influence scores with deduplication.

- **equal_weighting_justification_analysis.py**  
  Empirical + theoretical justification for influence metric weighting.

- **political_discourse_analyzer2.py**  
  Complete political discourse analyzer (blame, persuasion, influence).

---

## 📊 Methodology Overview

This system implements a **six-dimensional analysis model**:

1. **Causal Relationships** (regex-based attribution patterns)  
2. **Thematic Keywords** (15 political/social categories)  
3. **Persuasive Techniques** (8 techniques, 40+ sub-patterns)  
4. **Emotion Classification** (7 emotions, DistilRoBERTa)  
5. **Sentiment Analysis** (positive/negative polarity)  
6. **Topic Modeling** (TF-IDF key phrase extraction)

**Influence Intensity Formula**:  
Statistical validation is done via **bootstrap confidence intervals** (1,000 iterations) and **effect size analysis**.

For the full methodology, see:  
📄 [`docs/Methodology_Master_Synopsis.md`](docs/Methodology_Master_Synopsis.md)

---

## 🚀 How to Use

Clone the repo:
```bash
git clone https://github.com/syntax606/podcast-influence-analysis.git
cd podcast-influence-analysis
