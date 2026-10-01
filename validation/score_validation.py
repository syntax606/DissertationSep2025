#!/usr/bin/env python3
"""
Score completed annotation workbooks against the key: how well the dissertation's measures match human judgement.

    python validation/score_validation.py validation_round1 --annotators A B

Reports, for each task:
  emotion     agreement and Cohen's kappa, model vs each annotator and annotator vs annotator; per-emotion precision;
              corpus-weighted accuracy (strata re-weighted to their share of all segments)
  blame       precision of the blame detector (is it blame?), direct/meta agreement and kappa, target accuracy
  persuasion  precision per technique and overall (corpus-weighted), the figure the dissertation reported as > 0.82
  causal      precision per pattern and overall (corpus-weighted)
  recall      on random segments: recall and precision of the technique and causal detectors
Intervals are Wilson 95% for proportions. Items marked "unsure" are excluded and counted.

Writes validation_report.md and validation_scores.json into the folder.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    sys.exit("pip install openpyxl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_rules as R  # noqa: E402

SHEETS = {"emotion": "Emotion", "blame": "Blame", "persuasion": "Persuasion", "causal": "Causal", "recall": "Recall"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), n


def fmt(p, ci=None):
    if p != p:
        return "n/a"
    return f"{p:.0%}" + (f" [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "")


def read_labels(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    labels = {}
    for task, sheet in SHEETS.items():
        if sheet not in wb.sheetnames:
            continue
        rows = wb[sheet].iter_rows(values_only=True)
        header = [str(h) if h is not None else "" for h in next(rows)]
        for row in rows:
            rec = dict(zip(header, row))
            if rec.get("item_id"):
                labels[rec["item_id"]] = {k: (str(v).strip().lower() if isinstance(v, str) else v) for k, v in rec.items()}
    return labels


def norm(v):
    return None if v in (None, "", "unsure") else v


def score(key, ann):
    by_task = defaultdict(list)
    for k in key:
        by_task[k["task"]].append(k)
    names = list(ann)
    res, md = {}, []

    # --- emotion
    items = by_task["emotion"]
    if items:
        r = {}
        md.append("## Emotion (7 classes)\n")
        for a in names:
            pairs = [(k["pred"], norm(ann[a].get(k["item_id"], {}).get("emotion")), k["weight"]) for k in items]
            done = [(p, h, w) for p, h, w in pairs if h]
            agree = sum(p == h for p, h, _ in done)
            wacc = sum(w for p, h, w in done if p == h) / sum(w for _, _, w in done) if done else float("nan")
            kap, n = kappa([p for p, _, _ in done], [h for _, h, _ in done])
            per = {e: (sum(1 for p, h, _ in done if p == e and h == e), sum(1 for p, _, _ in done if p == e))
                   for e in sorted({p for p, _, _ in done})}
            r[a] = {"n": len(done), "agreement": agree / len(done) if done else None,
                    "agreement_ci": wilson(agree, len(done)), "corpus_weighted_accuracy": wacc, "kappa": kap,
                    "precision_by_predicted_emotion": {e: k / n for e, (k, n) in per.items() if n}}
            md.append(f"- Model vs {a}: agreement {fmt(agree / len(done) if done else float('nan'), wilson(agree, len(done)))} "
                      f"on {len(done)} segments (sample is balanced by emotion); corpus-weighted accuracy {fmt(wacc)}; "
                      f"kappa {kap:.2f}")
            md.append("  - precision by predicted emotion: " +
                      ", ".join(f"{e} {k}/{n}" for e, (k, n) in per.items()))
        if len(names) >= 2:
            a, b = names[:2]
            la = [norm(ann[a].get(k["item_id"], {}).get("emotion")) for k in items]
            lb = [norm(ann[b].get(k["item_id"], {}).get("emotion")) for k in items]
            kap, n = kappa(la, lb)
            agree = sum(1 for x, y in zip(la, lb) if x and y and x == y)
            r["inter_annotator"] = {"kappa": kap, "n": n, "agreement": agree / n if n else None}
            md.append(f"- {a} vs {b}: agreement {fmt(agree / n if n else float('nan'))}, kappa {kap:.2f} (n = {n})")
        res["emotion"] = r
        md.append("")

    # --- blame
    items = by_task["blame"]
    if items:
        r = {}
        md.append("## Blame attribution\n")
        for a in names:
            lab = [(k, ann[a].get(k["item_id"], {})) for k in items]
            isb = [(k, norm(l.get("is_blame"))) for k, l in lab if norm(l.get("is_blame"))]
            yes = sum(v == "yes" for _, v in isb)
            wprec = (sum(k["weight"] for k, v in isb if v == "yes") / sum(k["weight"] for k, _ in isb)) if isb else float("nan")
            typed = [(k["pred"], norm(l.get("blame_type"))) for k, l in lab
                     if norm(l.get("blame_type")) in ("direct", "meta") and k["pred"] in ("direct", "meta")]
            tagree = sum(p == h for p, h in typed)
            kap, n = kappa([p for p, _ in typed], [h for _, h in typed])
            tgt = [norm(l.get("target_correct")) for k, l in lab if norm(l.get("is_blame")) == "yes" and norm(l.get("target_correct"))]
            tyes = sum(v == "yes" for v in tgt)
            r[a] = {"n": len(isb), "is_blame_precision": yes / len(isb) if isb else None,
                    "is_blame_precision_ci": wilson(yes, len(isb)), "corpus_weighted_precision": wprec,
                    "type_agreement": tagree / len(typed) if typed else None, "type_kappa": kap, "type_n": n,
                    "target_accuracy": tyes / len(tgt) if tgt else None, "target_n": len(tgt)}
            md.append(f"- {a}: flagged instances that are blame {fmt(yes / len(isb) if isb else float('nan'), wilson(yes, len(isb)))} "
                      f"(n = {len(isb)}; corpus-weighted {fmt(wprec)})")
            md.append(f"  - direct/meta: model agrees {fmt(tagree / len(typed) if typed else float('nan'))}, kappa {kap:.2f} (n = {n})")
            md.append(f"  - target correct when it is blame: {fmt(tyes / len(tgt) if tgt else float('nan'), wilson(tyes, len(tgt)))}")
        if len(names) >= 2:
            a, b = names[:2]
            la = [norm(ann[a].get(k["item_id"], {}).get("blame_type")) for k in items]
            lb = [norm(ann[b].get(k["item_id"], {}).get("blame_type")) for k in items]
            kap, n = kappa(la, lb)
            r["inter_annotator_type_kappa"] = kap
            md.append(f"- {a} vs {b} on blame type (direct/meta/neither): kappa {kap:.2f} (n = {n})")
        res["blame"] = r
        md.append("")

    # --- precision tasks
    for task, field, strata_name in (("persuasion", "genuine_use", "technique"), ("causal", "causal_claim", "pattern")):
        items = by_task[task]
        if not items:
            continue
        r = {}
        md.append(f"## {task.capitalize()} precision\n")
        for a in names:
            lab = [(k, norm(ann[a].get(k["item_id"], {}).get(field))) for k in items]
            lab = [(k, v) for k, v in lab if v]
            per = defaultdict(lambda: [0, 0])
            for k, v in lab:
                per[k["stratum"]][1] += 1
                per[k["stratum"]][0] += v == "yes"
            # corpus-weighted: each stratum's precision weighted by its share of all hits
            pool = {k["stratum"]: k["pool_size"] for k in items}
            tot = sum(pool[s] for s in per)
            wprec = sum(pool[s] * (y / n) for s, (y, n) in per.items() if n) / tot if tot else float("nan")
            yes = sum(v == "yes" for _, v in lab)
            r[a] = {"n": len(lab), "precision_balanced": yes / len(lab) if lab else None,
                    "precision_corpus_weighted": wprec,
                    f"by_{strata_name}": {s: {"yes": y, "n": n, "precision": y / n, "ci": wilson(y, n)} for s, (y, n) in per.items()}}
            md.append(f"- {a}: corpus-weighted precision {fmt(wprec)} (balanced sample {fmt(yes / len(lab) if lab else float('nan'))}, n = {len(lab)})")
            for s, (y, n) in sorted(per.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
                md.append(f"  - {s}: {y}/{n} = {fmt(y / n, wilson(y, n))}")
        if len(names) >= 2:
            a, b = names[:2]
            kap, n = kappa([norm(ann[a].get(k["item_id"], {}).get(field)) for k in items],
                           [norm(ann[b].get(k["item_id"], {}).get(field)) for k in items])
            r["inter_annotator_kappa"] = kap
            md.append(f"- {a} vs {b}: kappa {kap:.2f} (n = {n})")
        res[task] = r
        md.append("")

    # --- recall on random segments
    items = by_task["recall"]
    if items:
        r = {}
        md.append("## Recall on random segments\n")
        for a in names:
            tp = fp = fn = 0
            ctp = cfp = cfn = 0
            n_done = 0
            for k in items:
                l = ann[a].get(k["item_id"], {})
                if all(norm(l.get(t)) is None for t in R.PERSUASIVE_TECHNIQUES):
                    continue
                n_done += 1
                for t in R.PERSUASIVE_TECHNIQUES:
                    h, p = norm(l.get(t)) == "yes", t in k["pred_techniques"]
                    tp += h and p
                    fp += p and not h
                    fn += h and not p
                hc, pc = norm(l.get("causal_claim")) == "yes", k["pred_causal"]
                ctp += hc and pc
                cfp += pc and not hc
                cfn += hc and not pc
            rec = tp / (tp + fn) if tp + fn else float("nan")
            prec = tp / (tp + fp) if tp + fp else float("nan")
            crec = ctp / (ctp + cfn) if ctp + cfn else float("nan")
            cprec = ctp / (ctp + cfp) if ctp + cfp else float("nan")
            r[a] = {"segments": n_done, "technique_recall": rec, "technique_precision": prec,
                    "technique_counts": {"tp": tp, "fp": fp, "fn": fn},
                    "causal_recall": crec, "causal_precision": cprec, "causal_counts": {"tp": ctp, "fp": cfp, "fn": cfn}}
            md.append(f"- {a} ({n_done} segments): techniques recall {fmt(rec, wilson(tp, tp + fn))}, precision "
                      f"{fmt(prec, wilson(tp, tp + fp))}; causal claims recall {fmt(crec, wilson(ctp, ctp + cfn))}, "
                      f"precision {fmt(cprec, wilson(ctp, ctp + cfp))}")
        res["recall"] = r
        md.append("")
    return res, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--annotators", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    folder = Path(args.folder)
    key = [json.loads(l) for l in open(folder / "key.jsonl", encoding="utf-8") if l.strip()]
    ann = {}
    for a in args.annotators:
        p = folder / f"annotation_{a}.xlsx"
        if p.exists():
            ann[a] = read_labels(p)
        else:
            print(f"(no workbook for {a} at {p}; skipped)")
    if not ann:
        sys.exit("No annotation workbooks found")
    res, md = score(key, ann)
    done = {a: sum(1 for v in labels.values() if any(norm(x) for k, x in v.items()
                                                       if k not in ("item_id", "before", "text", "after", "technique", "target", "note")))
            for a, labels in ann.items()}
    header = [f"# Validation report\n", f"Items in the sample: {len(key)}. Labelled: " +
              ", ".join(f"{a} {n}" for a, n in done.items()) + ".", ""]
    text = "\n".join(header + md)
    print(text)
    (folder / "validation_report.md").write_text(text + "\n")
    (folder / "validation_scores.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {folder / 'validation_report.md'} and validation_scores.json")


if __name__ == "__main__":
    main()
