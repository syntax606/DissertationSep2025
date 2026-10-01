#!/usr/bin/env python3
"""
Draw a blind, stratified sample of segments for human validation of the dissertation's measures.

It rebuilds the four validations the dissertation reports (Chapter 3), on the same data and with the same rules:

  task         what the annotator judges                                   default size
  emotion      the emotion expressed in a ~6-second segment (7 classes)     210 (30 per model-predicted emotion)
  blame        is it blame, direct or meta, and is the named target right   300 (all predicted types represented)
  persuasion   is each flagged term a real use of its technique             200 (25 per technique)
  causal       does each flagged connective introduce a causal claim        135 (15 per pattern)
  recall       which techniques / causal claims a random segment contains   150 random segments

Annotators never see the model's output, the show or the political orientation; those go into a separate key file.
Every item records its sampling stratum and weight, so the scorer can estimate corpus-level accuracy rather than
accuracy on the over-sampled strata.

Only the Python standard library is needed. Run it where the data is:

    export DISSERTATION_DATA_DIR="/Volumes/Caro Drive/podcast-dissertation-audio"
    python validation/make_validation_set.py --out validation_round1

Then build the annotation workbooks (needs openpyxl):

    python validation/build_workbooks.py validation_round1 --annotators A B

The output contains transcript excerpts, so it is git-ignored: keep it private.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_rules as R  # noqa: E402

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def orientation_of(name):
    if name.startswith("Tenet_"):
        return "Tenet"
    if name.startswith("Right wing__"):
        return "Right wing"
    if name.startswith("Left wing__"):
        return "Left wing"
    return None


class Reservoir:
    """Keeps a uniform random sample of fixed size from a stream, and counts the stream."""

    def __init__(self, k, rng):
        self.k, self.rng, self.n, self.items = k, rng, 0, []

    def add(self, item):
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self.items[j] = item


def mark(text, start, end):
    return text[:start] + "[[" + text[start:end] + "]]" + text[end:]


def predicted_emotion(seg):
    de = seg.get("dominant_emotion") or {}
    return de.get("label") or seg.get("emotion_label")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DISSERTATION_DATA_DIR"), help="local copy of the bucket")
    ap.add_argument("--out", required=True, help="output folder (keep private)")
    ap.add_argument("--episodes-per-orientation", type=int, default=80,
                    help="episodes sampled from each of Tenet, right-wing, left-wing (default 80)")
    ap.add_argument("--blame-episodes-per-orientation", type=int, default=200,
                    help="episodes scanned for blame (blame is rare inside one segment, so more are needed; default 200)")
    ap.add_argument("--emotion-per-class", type=int, default=30)
    ap.add_argument("--blame", type=int, default=300)
    ap.add_argument("--persuasion-per-technique", type=int, default=25)
    ap.add_argument("--causal-per-pattern", type=int, default=15)
    ap.add_argument("--recall", type=int, default=150)
    ap.add_argument("--min-chars", type=int, default=40, help="skip segments shorter than this")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not args.data_dir:
        sys.exit("Set DISSERTATION_DATA_DIR or pass --data-dir")
    src = Path(args.data_dir) / "enriched_transcripts"
    out = Path(args.out)
    if out.resolve().is_relative_to(Path(args.data_dir).resolve()):
        sys.exit("Choose an output folder outside the data folder")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    by_orient = defaultdict(list)
    for name in sorted(os.listdir(src)):
        if name.endswith("_enriched.json") and (o := orientation_of(name)):
            by_orient[o].append(name)
    episodes, blame_only = [], []
    for o in ("Tenet", "Right wing", "Left wing"):
        pool = list(by_orient[o])
        rng.shuffle(pool)
        main = pool[:args.episodes_per_orientation]
        extra = pool[args.episodes_per_orientation:max(args.episodes_per_orientation, args.blame_episodes_per_orientation)]
        episodes += [(o, n, False) for n in main] + [(o, n, True) for n in extra]
    print(f"Reading {len(episodes)} episodes ({sum(1 for *_, b in episodes if not b)} for all tasks, the rest for blame "
          "only): " + ", ".join(f"{o} {sum(1 for x, *_ in episodes if x == o)}" for o in ("Tenet", "Right wing", "Left wing")))

    big = 4  # reservoirs hold more than needed so strata can be balanced across orientations afterwards
    emo = {e: Reservoir(args.emotion_per_class * big, rng) for e in EMOTIONS}
    blame = defaultdict(lambda: Reservoir(args.blame * big, rng))
    pers = {t: Reservoir(args.persuasion_per_technique * big, rng) for t in R.PERSUASIVE_TECHNIQUES}
    caus = {p: Reservoir(args.causal_per_pattern * big, rng) for p, _ in R.COMPILED_CAUSAL}
    rand = Reservoir(args.recall * big, rng)
    totals = Counter()

    for i, (orient, name, only_blame) in enumerate(episodes, 1):
        with open(src / name, encoding="utf-8") as f:
            segs = json.load(f).get("segments", [])
        texts = [s.get("text", "").strip() for s in segs]
        for idx, (seg, text) in enumerate(zip(segs, texts)):
            if len(text) < args.min_chars:
                continue
            totals["blame_segments"] += 1
            totals["segments"] += not only_blame
            base = {"file": name, "orientation": orient, "segment": idx,
                    "before": texts[idx - 1] if idx > 0 else "", "after": texts[idx + 1] if idx + 1 < len(texts) else ""}
            for b in R.blame_instances(text):
                blame[b["type"]].add({**base, "text": mark(text, b["trigger_start"], b["trigger_end"]),
                                      "pred": b["type"], "pred_target": b["entity"], "trigger": b["trigger"],
                                      "pred_detail": b})
            if only_blame:
                continue
            e = predicted_emotion(seg)
            if e in emo:
                emo[e].add({**base, "text": text, "pred": e, "pred_scores": seg.get("emotion_scores")})
            for tech, term, s0, s1 in R.persuasive_hits(text):
                pers[tech].add({**base, "text": mark(text, s0, s1), "pred": tech, "term": term})
            for pat, s0, s1 in R.causal_hits(text):
                caus[pat].add({**base, "text": mark(text, s0, min(s1, s0 + len(pat) + 1)), "pred": pat})
            rand.add({**base, "text": text,
                      "pred_techniques": sorted({h[0] for h in R.persuasive_hits(text)}),
                      "pred_causal": bool(R.causal_hits(text))})
        if i % 50 == 0:
            print(f"  read {i}/{len(episodes)} episodes")

    def balanced(res, k):
        """Draw k items from a reservoir, as evenly across orientations as the pool allows."""
        groups = defaultdict(list)
        for it in res.items:
            groups[it["orientation"]].append(it)
        for g in groups.values():
            rng.shuffle(g)
        chosen = []
        while len(chosen) < k and any(groups.values()):
            for o in ("Tenet", "Right wing", "Left wing"):
                if groups.get(o) and len(chosen) < k:
                    chosen.append(groups[o].pop())
        return chosen

    items, key = [], []

    def emit(task, stratum, pool_size, chosen):
        w = pool_size / len(chosen) if chosen else 0
        for it in chosen:
            item_id = f"{task[:3]}-{rng.getrandbits(32):08x}"
            items.append({"item_id": item_id, "task": task, "before": it["before"], "text": it["text"],
                          "after": it["after"], **({"term": it["term"], "technique": it["pred"]} if task == "persuasion" else {}),
                          **({"target": it["pred_target"]} if task == "blame" else {})})
            key.append({"item_id": item_id, "task": task, "stratum": stratum, "pool_size": pool_size, "weight": w,
                        **{k: v for k, v in it.items() if k not in ("before", "after")}})

    for e in EMOTIONS:
        emit("emotion", e, emo[e].n, balanced(emo[e], args.emotion_per_class))
    # Blame: every predicted type gets a share (meta and unclear are rare), the rest proportional to frequency.
    types = sorted(blame)
    floor = min(60, args.blame // max(1, len(types)))
    alloc = {t: min(floor, blame[t].n) for t in types}
    rest = args.blame - sum(alloc.values())
    total_n = sum(blame[t].n for t in types) or 1
    for t in types:
        alloc[t] += min(len(blame[t].items) - alloc[t], round(rest * blame[t].n / total_n))
    for t in types:
        emit("blame", t, blame[t].n, balanced(blame[t], alloc[t]))
    for t in R.PERSUASIVE_TECHNIQUES:
        emit("persuasion", t, pers[t].n, balanced(pers[t], args.persuasion_per_technique))
    for p in caus:
        emit("causal", p, caus[p].n, balanced(caus[p], args.causal_per_pattern))
    emit("recall", "random", rand.n, balanced(rand, args.recall))

    rng.shuffle(items)  # so the order carries no information about strata
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "key.jsonl", "w", encoding="utf-8") as f:
        for k in key:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    summary = {
        "seed": args.seed, "episodes": sum(1 for *_, b in episodes if not b), "segments_read": totals["segments"],
        "items": Counter(i["task"] for i in items),
        "pools": {"emotion": {e: emo[e].n for e in EMOTIONS}, "blame": {t: blame[t].n for t in types},
                  "persuasion": {t: pers[t].n for t in pers}, "causal": {p: caus[p].n for p in caus},
                  "recall": rand.n},
        "episode_files": [n for _, n, b in episodes if not b], "blame_only_episode_files": [n for _, n, b in episodes if b],
        "segments_read_for_blame": totals["blame_segments"],
    }
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(items)} items written to {out}/ (items.jsonl for annotators, key.jsonl kept back)")
    for t, n in summary["items"].items():
        print(f"  {t:<11} {n}")
    short = [f"{t}:{s}" for t, d in summary["pools"].items() if isinstance(d, dict) for s, n in d.items()
             if (t == "emotion" and n < args.emotion_per_class) or (t == "persuasion" and n < args.persuasion_per_technique)
             or (t == "causal" and n < args.causal_per_pattern)]
    if short:
        print("  pools smaller than requested (raise --episodes-per-orientation): " + ", ".join(short))


if __name__ == "__main__":
    main()
