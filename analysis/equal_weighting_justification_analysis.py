#!/usr/bin/env python3
"""
Compare weighting schemes for the influence intensity metric on the real episode data.

Rewritten. The published version could not find the Tenet episodes, so it generated 20 synthetic "Foreign" episodes
from assumed values (np.random.normal around 3.2 causal and 2.25 persuasive per 1k words) and compared them with 500
left-wing episodes and no right-wing ones. Its results therefore say nothing about the real data, and they did not
match Table 3.2 of the dissertation either (equal weighting F = 7.06, causal-only F = 11.64, r = -0.08).

This version:
- reads every episode's causal and persuasive densities from the enhanced-analysis batch files (the same source as
  the published influence scores), de-duplicated by source file, for all three groups: Tenet, right-wing, left-wing;
- scores each episode as  w * causal_density + (1 - w) * persuasive_density  for each scheme;
- reports, per scheme: one-way ANOVA across the three groups (F, p, eta squared), the Kruskal-Wallis test (the scores
  are skewed), Cohen's d for each pair of groups, and group means;
- reports the correlation between the two components;
- generates nothing: if a group is missing it stops and says so.

Note: the published formula, (causal + persuasive) / words x 1000, is exactly twice the equal-weighted score, so it
ranks episodes identically and gives the same test statistics as the 0.5/0.5 row.

Usage:
    export DISSERTATION_DATA_DIR="/Volumes/Caro Drive/podcast-dissertation-audio"
    python analysis/equal_weighting_justification_analysis.py --output-dir outputs/weighting
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage_backend import storage  # noqa: E402

PROJECT_ID = "handy-vortex-459018-g0"
BUCKET_NAME = "podcast-dissertation-audio"

SCHEMES = {
    "Causal-only (1.0, 0.0)": 1.0,
    "Causal-heavy (0.7, 0.3)": 0.7,
    "Variance-optimized (0.6, 0.4)": 0.6,
    "Equal (0.5, 0.5)": 0.5,
    "Persuasive-heavy (0.3, 0.7)": 0.3,
    "Persuasive-only (0.0, 1.0)": 0.0,
}
GROUPS = ("Tenet", "Right wing", "Left wing")


def orientation_of(blob_name):
    base = blob_name.split("/")[-1]
    if base.lower().startswith("tenet"):
        return "Tenet"
    if base.startswith("Right_wing_"):
        return "Right wing"
    if base.startswith("Left_wing_"):
        return "Left wing"
    return None


def load_episodes():
    bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME)
    episodes, seen, duplicates, files = [], set(), 0, 0
    for prefix in ("tenet_enhanced_analysis/", "us_podcast_enhanced_analysis/"):
        for blob in bucket.list_blobs(prefix=prefix):
            if not (blob.name.endswith(".json") and "batch" in blob.name):
                continue
            orientation = orientation_of(blob.name)
            if orientation is None:
                continue
            files += 1
            for ep in json.loads(blob.download_as_string()):
                src = ep.get("metadata", {}).get("source_file", "")
                if not src or src in seen:
                    duplicates += 1
                    continue
                seen.add(src)
                m = ep.get("influence_metrics", {})
                if "causal_density_per_1k" not in m or "persuasive_density_per_1k" not in m:
                    continue
                episodes.append({
                    "orientation": orientation,
                    "causal": float(m["causal_density_per_1k"]),
                    "persuasive": float(m["persuasive_density_per_1k"]),
                    "influence_ratio": float(m.get("influence_ratio", np.nan)),
                    "words": int(ep.get("metadata", {}).get("total_word_count", 0)),
                })
    return episodes, files, duplicates


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else float("nan")


def evaluate(episodes):
    by = {g: [e for e in episodes if e["orientation"] == g] for g in GROUPS}
    causal = np.array([e["causal"] for e in episodes])
    persuasive = np.array([e["persuasive"] for e in episodes])
    results = {}
    for name, w in SCHEMES.items():
        scores = {g: np.array([w * e["causal"] + (1 - w) * e["persuasive"] for e in by[g]]) for g in GROUPS}
        f, p = stats.f_oneway(*scores.values())
        allv = np.concatenate(list(scores.values()))
        grand = allv.mean()
        ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in scores.values())
        ss_total = ((allv - grand) ** 2).sum()
        h, hp = stats.kruskal(*scores.values())
        results[name] = {
            "weights": [w, round(1 - w, 1)],
            "anova_F": float(f), "anova_p": float(p), "eta_squared": float(ss_between / ss_total),
            "kruskal_H": float(h), "kruskal_p": float(hp),
            "group_means": {g: float(v.mean()) for g, v in scores.items()},
            "group_sd": {g: float(v.std(ddof=1)) for g, v in scores.items()},
            "cohens_d": {
                "Right vs Left": cohens_d(scores["Right wing"], scores["Left wing"]),
                "Tenet vs Right": cohens_d(scores["Tenet"], scores["Right wing"]),
                "Tenet vs Left": cohens_d(scores["Tenet"], scores["Left wing"]),
                "Tenet vs US": cohens_d(scores["Tenet"], np.concatenate([scores["Right wing"], scores["Left wing"]])),
            },
        }
    r, rp = stats.pearsonr(causal, persuasive)
    rho, rhop = stats.spearmanr(causal, persuasive)
    return results, {"pearson_r": float(r), "pearson_p": float(rp), "spearman_rho": float(rho), "spearman_p": float(rhop)}


def report(results, corr, counts, files, duplicates):
    lines = [
        "WEIGHTING SCHEMES FOR INFLUENCE INTENSITY: REAL DATA",
        "=" * 60,
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
        f"Episodes: " + ", ".join(f"{g} {counts[g]}" for g in GROUPS) + f"  (from {files} batch files; {duplicates} duplicates skipped)",
        "Score per episode = w x causal density + (1 - w) x persuasive density (both per 1,000 words).",
        "",
        f"Correlation between components: Pearson r = {corr['pearson_r']:.3f} (p = {corr['pearson_p']:.2g}), "
        f"Spearman rho = {corr['spearman_rho']:.3f}",
        "",
        f"{'Scheme':<32}{'F':>9}{'eta2':>8}{'H':>9}  {'d R-L':>7}{'d T-R':>7}{'d T-L':>7}{'d T-US':>8}",
    ]
    for name, r in results.items():
        d = r["cohens_d"]
        lines.append(f"{name:<32}{r['anova_F']:>9.1f}{r['eta_squared']:>8.3f}{r['kruskal_H']:>9.1f}  "
                     f"{d['Right vs Left']:>7.2f}{d['Tenet vs Right']:>7.2f}{d['Tenet vs Left']:>7.2f}{d['Tenet vs US']:>8.2f}")
    best = max(results, key=lambda k: results[k]["anova_F"])
    lines += [
        "",
        f"Highest F: {best}.",
        "All p-values: " + ", ".join(f"{k.split(' (')[0]} {v['anova_p']:.2g}" for k, v in results.items()),
        "",
        "Group means of the equal-weighted score (x2 = the published influence intensity):",
    ]
    eq = results["Equal (0.5, 0.5)"]
    for g in GROUPS:
        lines.append(f"  {g:<11} {eq['group_means'][g]:.3f}  (x2 = {2 * eq['group_means'][g]:.3f}), SD {eq['group_sd'][g]:.3f}")
    lines += [
        "",
        "Reading: F measures how well a scheme separates the three groups. A higher F is not by itself a reason to",
        "prefer a scheme; the published metric is an unweighted sum, which is equivalent to the 0.5/0.5 row.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare influence-metric weighting schemes on the real data")
    parser.add_argument("--output-dir", default="outputs/weighting")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    episodes, files, duplicates = load_episodes()
    counts = Counter(e["orientation"] for e in episodes)
    missing = [g for g in GROUPS if counts[g] < 2]
    if missing:
        sys.exit(f"Not enough episodes for {', '.join(missing)} ({dict(counts)}). Nothing is generated to fill the gap.")

    results, corr = evaluate(episodes)
    text = report(results, corr, counts, files, duplicates)
    print(text)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"weighting_real_data_{stamp}.json"), "w") as f:
        json.dump({"episodes": dict(counts), "batch_files": files, "duplicates_skipped": duplicates,
                   "component_correlation": corr, "schemes": results}, f, indent=2)
    with open(os.path.join(args.output_dir, f"weighting_real_data_{stamp}.txt"), "w") as f:
        f.write(text + "\n")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(results)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh([n.split(" (")[0] for n in names][::-1], [results[n]["anova_F"] for n in names][::-1], color="#4a6fa5")
        ax.set_xlabel("ANOVA F across Tenet, right-wing and left-wing episodes")
        ax.set_title(f"Weighting schemes on real data (n = {len(episodes):,})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"weighting_real_data_{stamp}.png"), dpi=150)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
