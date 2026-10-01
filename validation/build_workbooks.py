#!/usr/bin/env python3
"""
Turn a validation sample (from make_validation_set.py) into one annotation workbook per annotator.

    python validation/build_workbooks.py validation_round1 --annotators A B

Each workbook has an Instructions sheet and one sheet per task. Annotators fill only the yellow columns, using the
drop-down lists. They never see the model's predictions, the show or the orientation. Two independent annotators
are recommended: the scorer then reports agreement between people as well as between people and the model.

Needs openpyxl (pip install openpyxl).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError:
    sys.exit("pip install openpyxl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_rules as R  # noqa: E402

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
YN = ["yes", "no", "unsure"]
TECHS = list(R.PERSUASIVE_TECHNIQUES)

INPUT_FILL = PatternFill("solid", fgColor="FFF4C2")
HEAD_FILL = PatternFill("solid", fgColor="DDE3EA")
WRAP = Alignment(wrap_text=True, vertical="top")

# (sheet, task, columns the annotator fills: (header, allowed values or None for free text, width))
TASKS = [
    ("Emotion", "emotion", [("emotion", EMOTIONS, 12), ("second_emotion (optional)", EMOTIONS + ["none"], 14)]),
    ("Blame", "blame", [("is_blame", YN, 10), ("blame_type", ["direct", "meta", "neither"], 12),
                        ("target_correct", YN, 12)]),
    ("Persuasion", "persuasion", [("genuine_use", YN, 12)]),
    ("Causal", "causal", [("causal_claim", YN, 12)]),
    ("Recall", "recall", [(t, ["yes", "no"], 11) for t in TECHS] + [("causal_claim", ["yes", "no"], 11)]),
]

INSTRUCTIONS = f"""HOW TO ANNOTATE

Work through each sheet. Fill only the yellow columns, using the drop-down. Add a note if something is unclear.
Judge what the speaker is doing in the segment, using the text before and after for context. The segments are
automatic transcripts of podcasts (about 6 seconds of speech each), so expect missing punctuation and some
transcription errors. Do not look anything up and do not discuss items with the other annotator.

EMOTION: Which one emotion does the middle segment ("text") express most strongly?
  {", ".join(EMOTIONS)}. Choose "neutral" when no emotion is clearly expressed. A second emotion is optional.

BLAME: The phrase in [[double brackets]] was flagged as possible blame, and "target" is the person or group the
  tool linked it to.
  is_blame: does the speaker assign responsibility for something bad to someone?
  blame_type: direct = the speaker blames someone ("Biden caused inflation").
              meta = the speaker comments on blaming ("they always blame the media", "the blame game").
              neither = not blame (e.g. "due to the weather", a neutral "because of").
  target_correct: is "target" the one being blamed (or, for meta-blame, the one doing or receiving the blaming)?

PERSUASION: The term in [[double brackets]] was flagged as the technique named in "technique".
  genuine_use: is the speaker actually using that technique here, as defined below? ("according to the schedule" is
  not an authority appeal; "the constitution says" in a legal explanation may not be a patriotic appeal.)

CAUSAL: The connective in [[double brackets]] was flagged as a causal pattern.
  causal_claim: does it introduce a claim that one thing caused or led to another? ("so that" usually expresses
  purpose, not cause.)

RECALL: Random segments, nothing flagged. Mark yes for every technique that is present, and whether the segment
  contains a causal claim. Mark no otherwise.

TECHNIQUES
""" + "\n".join(f"  {t}: {d}" for t, d in R.TECHNIQUE_DEFINITIONS.items())


def build(items, path, annotator):
    wb = Workbook()
    ws = wb.active
    ws.title = "Instructions"
    ws["A1"] = f"Annotator: {annotator}"
    ws["A1"].font = Font(bold=True, size=13)
    for i, line in enumerate(INSTRUCTIONS.splitlines(), start=3):
        ws.cell(row=i, column=1, value=line)
    ws.column_dimensions["A"].width = 130

    for sheet, task, inputs in TASKS:
        rows = [it for it in items if it["task"] == task]
        if not rows:
            continue
        ws = wb.create_sheet(sheet)
        fixed = [("item_id", 13), ("before", 38), ("text", 60), ("after", 38)]
        if task == "persuasion":
            fixed += [("technique", 18)]
        if task == "blame":
            fixed += [("target", 16)]
        headers = fixed + [(h, w) for h, _, w in inputs] + [("note", 30)]
        for c, (h, w) in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = Font(bold=True)
            cell.fill = HEAD_FILL
            ws.column_dimensions[cell.column_letter].width = w
        ws.freeze_panes = "B2"
        first_input = len(fixed) + 1
        for r, it in enumerate(rows, start=2):
            for c, (h, _) in enumerate(fixed, start=1):
                ws.cell(row=r, column=c, value=it.get(h, "")).alignment = WRAP
            for c in range(first_input, len(headers) + 1):
                ws.cell(row=r, column=c).fill = INPUT_FILL
            ws.row_dimensions[r].height = 75
        for j, (h, allowed, _) in enumerate(inputs):
            if allowed:
                col = ws.cell(row=1, column=first_input + j).column_letter
                dv = DataValidation(type="list", formula1='"' + ",".join(allowed) + '"', allow_blank=True)
                dv.add(f"{col}2:{col}{len(rows) + 1}")
                ws.add_data_validation(dv)
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="output folder of make_validation_set.py")
    ap.add_argument("--annotators", nargs="+", default=["A", "B"])
    args = ap.parse_args()
    folder = Path(args.folder)
    items = [json.loads(l) for l in open(folder / "items.jsonl", encoding="utf-8") if l.strip()]
    for a in args.annotators:
        path = folder / f"annotation_{a}.xlsx"
        build(items, path, a)
        print(f"wrote {path}")
    print("Give each annotator only their own workbook. Keep key.jsonl to yourself until scoring.")


if __name__ == "__main__":
    main()
