# Human validation kit

Recreates the dissertation's manual validation (Chapter 3) on the same data, with the same rules. Before this kit,
the reported figures could not be checked: no labelled data was found. The figures were 85% agreement on emotion,
89% / κ = 0.84 on blame, and more than 0.82 precision on persuasive techniques.

| Task | Question for the annotator | Default sample | Reported as |
|---|---|---|---|
| Emotion | Which of 7 emotions does the segment express? | 210 segments, 30 per model-predicted emotion | agreement and Cohen's κ with the model; κ between annotators; accuracy re-weighted to the corpus |
| Blame | Is it blame? Direct or meta? Is the target right? | 300 detector hits, every predicted type represented | precision of the detector; direct/meta agreement and κ; target accuracy |
| Persuasion | Is the flagged term a real use of its technique? | 200 hits, 25 per technique | precision per technique and overall (re-weighted to how often each fires) |
| Causal | Does the flagged connective introduce a causal claim? | 135 hits, 15 per pattern | precision per pattern and overall |
| Recall | Which techniques and causal claims does a random segment contain? | 150 random segments | recall and precision of the detectors |

The rules being tested are in `reference_rules.py`: the persuasive and causal lexicons behind the influence score,
and the blame classifier from `no_sarcasm.py`. They are kept exactly as they were used, including their known quirks.

## Running it

```bash
export DISSERTATION_DATA_DIR="/path/to/podcast-dissertation-audio"

# 1. Draw the sample (standard library only; reads about 600 episodes, so allow 10-20 minutes)
python validation/make_validation_set.py --out validation_round1

# 2. One workbook per annotator (pip install openpyxl)
python validation/build_workbooks.py validation_round1 --annotators A B

# 3. Annotators fill the yellow columns in their own workbook, independently

# 4. Score
python validation/score_validation.py validation_round1 --annotators A B
```

**What the annotators see.** Each item shows the segment and the segments either side. They do not see the model's
output, the show or the political orientation. Those go in `key.jsonl`, which only the scorer reads.

**Strata and weights.** Every item records its stratum and how many segments in the sample it stands for. The scorer
reports results on the balanced sample and re-weighted to the corpus.

**Two annotators recommended.** With two, κ between people gives the ceiling the model can reasonably be held to.
One annotator still gives model-vs-human figures.

**Allow about 5–7 hours per annotator** for the default 995 items. Every sample size is an option on
`make_validation_set.py`.

**Blame is rare.** In round 1 (seed 2026), the detector fired 257 times in 292,000 segments from 600 episodes (221
direct, 29 unclear, 7 meta). All 257 went into the sample, so the blame figures cover every instance found, not a
subsample. Round 1 has 912 items in total.

## Privacy

The output folder holds transcript excerpts. It is git-ignored: keep it private, and share only the scores.

## Not included

**Comparison with Da San Martino et al. (2019).** The dissertation reports precision against this annotated
propaganda corpus. The corpus has its own licence and is not distributed here. With a copy, the persuasive lexicon
can be run over its spans for that comparison.
