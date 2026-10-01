# Corrections and repairs

This copy of `DissertationSep2025` fixes the scripts so they run, and replaces two validation analyses that were not
measurements with real ones. The original repository and the original data on the external drive were not changed.

Everything below was checked against the data on the drive (`/Volumes/Caro Drive/podcast-dissertation-audio`) on
30 September 2026.

## The headline result reproduces exactly

`analysis/transparent_bootstrap.py` (output in `data/reproduction/`), run on the enhanced-analysis files, gives the Chapter 4 figures to three
decimal places:

| Group | Episodes | Influence intensity (95% CI) |
|---|---|---|
| Tenet Media | 405 | 5.454 [5.218, 5.700] |
| US right-wing | 3,741 | 5.684 [5.630, 5.737] |
| US left-wing | 2,291 | 4.782 [4.720, 4.844] |

Re-scoring 25 Tenet transcripts from scratch with `tenet_enhanced_analysis.py` gives the same per-episode scores that
are stored in the enhanced-analysis files.

## Scripts repaired

| Script | Problem | Fix |
|---|---|---|
| all scripts | Hard-wired to the Google Cloud bucket | `storage_backend.py`: set `DISSERTATION_DATA_DIR` to run on a local copy (the drive). Outputs go to `DISSERTATION_OUTPUT_DIR` (default `./outputs`), never into the data folder. Without the variable, the scripts use the bucket as before. |
| `enrichment_pipeline.py` | Would not run: curly quotes, lost indentation and markdown fences pasted into the code | Rewritten with the same logic, models and keyword lists. It reproduces the stored `persuasive_techniques`, `detected_topics` and `topic_scores` for all 2,491 enriched segments checked. |
| `recalculate_influence_scores.py` | Crashed on every run: it looked up `show_name` on records that don't have it | Fixed. Also seeded the bootstrap (it was unseeded), with `--seed 42` as the default. |
| `bootstrap_confidence_intervals.py` | Crashed if the output folder didn't exist. It also computes a *simplified* score, not the dissertation's (7.45 vs 5.00 on the same 25 Tenet episodes). | Fixed the folder, and labelled the script. The published intervals come from `transparent_bootstrap.py`, now included. |
| `transparent_bootstrap.py`, `per_show_influence_bootstrap.py` | Behind the published intervals and the per-show table, but missing from the repo | Added, with local-data support. |
| `meantime_transcribe.py` | VM-only paths. A public Healthchecks ping URL and ntfy topic. A crash when a podcast had no audio. | Paths are options. Alerts are optional environment variables. Crash fixed. |
| `political_discourse_analyzer2.py`, `tenet_enhanced_analysis.py`, `us_podcast_analysis_enhanced_timeline.py` | Cloud-only | Run locally; logic unchanged. |

**README.** The README linked a `LICENSE` and a `requirements.txt` that weren't there; both now exist. The example
command used an option (`--bootstrap`) that the script doesn't have.

## Weighting schemes: now measured on the real data

**The original analysis.** `equal_weighting_justification_analysis.py` couldn't find the Tenet episodes.
- It generated 20 synthetic "Foreign" episodes from assumed values and compared them with 500 left-wing episodes. No
  right-wing episodes were included.
- Its saved output (`data/testing/weighting_justification.json`) does not match Table 3.2 either. Equal weighting
  scored F = 7.06, the best scheme was causal-only (F = 11.64), and the component correlation was r = −0.08.

**The new analysis.** The rewritten script uses every episode (Tenet 405, right-wing 3,741, left-wing 2,291),
de-duplicated as in the published analysis:

| Scheme (causal, persuasive) | ANOVA F | η² | d right vs left | d Tenet vs right | d Tenet vs left |
|---|---|---|---|---|---|
| Causal only (1.0, 0.0) | 79.0 | 0.024 | 0.31 | −0.38 | −0.08 |
| Causal-heavy (0.7, 0.3) | 140.5 | 0.042 | 0.45 | −0.27 | 0.16 |
| (0.6, 0.4) | 173.4 | 0.051 | 0.51 | −0.21 | 0.28 |
| **Equal (0.5, 0.5)** | **208.8** | **0.061** | **0.56** | **−0.13** | **0.40** |
| Persuasive-heavy (0.3, 0.7) | 253.6 | 0.073 | 0.63 | 0.05 | 0.60 |
| Persuasive only (0.0, 1.0) | 212.7 | 0.062 | 0.55 | 0.27 | 0.69 |

**Component correlation:** Pearson r = 0.009 (p = 0.48). All p-values are below 1e-30.

What this means for Section 3.4 and Table 3.2:
- **Equal weighting is not the scheme that separates the groups best.** Persuasive-heavy does (F = 253.6).
- **The effect sizes are small to medium, not large.** Cohen's d is 0.56 right vs left, and 0.13 between Tenet and
  right-wing.
- **The two components are essentially uncorrelated (r ≈ 0.01), not r = 0.23.** That still supports treating them
  as distinct dimensions.
- **Equal weighting is still a reasonable choice, but only on interpretability grounds.** The published metric is an
  unweighted sum, so its ranking is the same as the 0.5/0.5 row. Twice the equal-weighted group means reproduces the
  headline figures (5.454, 5.683, 4.781).

Output: `data/reproduction/weighting_real_data_20260930_153249.json` and `.txt`.

## Processing time: now measured

**The original analysis.** It reported 0.54 s per episode. It came from `computational_performance_analysis.py`,
which timed `time.sleep()` calls, not the pipeline, and it did not include transcription.

**The new analysis.** The rewritten script times the real code. Measured on an Apple M3 iMac (8 cores), Python 3.13:

| Stage | Measured |
|---|---|
| Scoring (causal, keyword and persuasive analysis) | median 0.031 s per episode (mean 0.042 s; 100 episodes averaging 10,214 words). Reading the transcript file adds a median 0.21 s. |
| Emotion and sentiment models | 12 ms per segment (emotion 6.3 ms, sentiment 5.9 ms): a median 8.4 s per episode over 3 episodes (3,689 segments), after 63 s of model loading |
| Sentence-BERT embeddings | not measured: `sentence-transformers` is not installed in `dissertation_env` |
| Whisper transcription | not measured: `openai-whisper` is not installed. Run with `--audio <files>` to measure it. |

So the full pipeline takes several seconds per episode once the models run, plus transcription, which will dominate.
"0.54 seconds per episode" and "real-time from raw audio" are not supported.

Output: `data/reproduction/performance_20260930_153542.json` and `.txt`.

## Claims that could not be checked

- **Manual validation.** The dissertation reports 85% agreement on 200 emotion segments, 89% and κ = 0.84 on 301
  blame segments, and more than 0.82 precision against Da San Martino et al. On the drive:
  - the validation sheet (`Testing_output/methodology_validation_validation_sample_20250811_102259.csv`, 60 episodes)
    has every manual column empty;
  - the sarcasm annotation set (`tmp/sarcasm_candidates.csv`, `tmp/sarcasm_controls.csv`, 300 segments) has empty
    labels;
  - no other annotation files were found.

  If the labels exist elsewhere, adding them here would settle it. Otherwise, `validation/` now holds a kit to
  redo the validation on the same data: it draws a blind sample, builds annotation workbooks and scores them.
- **Corpus size (Table 3.1).** The Tenet summary file gives 1,137,237 words for 405 episodes, not 5.9M. The show
  files cover 31 shows (15 right-wing, 15 left-wing, Tenet), not 91.
- **The F = 847.3 in the synopsis** does not appear in any output found.
- **"Tenet 2.2 techniques per episode" in `dissertation_key_findings.md`** does not match Table 4.1 (Tenet 4.8
  persuasive instances per episode). It appears to come from a different script and is not defined.

## Not changed

- **The data files under `data/`** are kept as they were, including the invalid `weighting_justification.json` and
  `computational_performance_formatted.txt`, so the record is complete. This file says which to trust.
- **`data/samples/sample_output.json`** (41 MB) was too large to copy here; it is unchanged in the original
  repository.
- **Every original script on the drive** is untouched.
