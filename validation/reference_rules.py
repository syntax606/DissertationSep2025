"""The rules being validated, copied from the scripts that produced the dissertation's numbers.

- CAUSAL_PATTERNS and PERSUASIVE_TECHNIQUES: tenet_enhanced_analysis.py / us_podcast_analysis_enhanced_timeline.py
  (the influence intensity score).
- Blame detection and the direct/meta classifier: no_sarcasm.py on the drive (the script behind
  blame_classification_final/). It is not in the repository, so its rules are reproduced here verbatim.

Kept separate so the validation tests exactly what was used, not a later improved version.
"""

import re

CAUSAL_PATTERNS = [
    r"because\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"due to\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"as a result of\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"resulting in\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"so that\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"therefore\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"consequently\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"which led to\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
    r"causing\s+(.+?)(?:[.,;!?]|\s+and|\s+but|$)",
]
COMPILED_CAUSAL = [(p.split("\\s+")[0], re.compile(p, re.IGNORECASE)) for p in CAUSAL_PATTERNS]

PERSUASIVE_TECHNIQUES = {
    "appeal_to_fear": ["threat", "danger", "they will come for you", "chaos", "destroy",
                       "invasion", "crisis", "collapse", "catastrophe", "disaster"],
    "emotional_language": ["outrage", "disgusting", "evil", "innocent", "perverted", "sinful",
                           "betrayal", "corrupt", "heinous", "vile", "shocking"],
    "us_vs_them": ["they want you to", "they are lying", "wake up", "sheeple", "we must fight",
                   "real americans", "patriots", "enemies", "traitors", "establishment"],
    "authority_reference": ["experts say", "the science is clear", "officials stated", "studies show",
                            "research proves", "data shows", "according to", "sources confirm"],
    "repetition_emphasis": ["again and again", "over and over", "time and time again",
                            "repeatedly", "constantly", "continuously"],
    "scapegoating": ["blame", "fault of", "responsible for our problems", "caused by",
                     "to blame for", "the reason for"],
    "patriotic_appeal": ["america first", "our country", "real patriots", "defend freedom",
                         "american values", "founding fathers", "constitution", "liberty"],
    "victimization": ["under attack", "being silenced", "persecution", "targeted",
                      "oppressed", "censored", "cancelled"],
}

TECHNIQUE_DEFINITIONS = {
    "appeal_to_fear": "Tries to make the listener afraid of a threat or consequence to win agreement.",
    "emotional_language": "Uses morally or emotionally loaded words to provoke a reaction rather than describe.",
    "us_vs_them": "Divides people into an in-group and a hostile or deceitful out-group.",
    "authority_reference": "Uses an authority (experts, studies, officials) to support a claim.",
    "repetition_emphasis": "Stresses that something happens repeatedly to amplify it.",
    "scapegoating": "Places blame for a problem on a person or group.",
    "patriotic_appeal": "Appeals to national identity, values or symbols to win agreement.",
    "victimization": "Presents the speaker or their side as attacked, silenced or persecuted.",
}


def persuasive_hits(text):
    """(technique, term, start, end) exactly as the dissertation counted them (substring for phrases)."""
    low = text.lower()
    hits = []
    for tech, terms in PERSUASIVE_TECHNIQUES.items():
        for term in terms:
            if " " in term:
                start = 0
                while (i := low.find(term, start)) != -1:
                    hits.append((tech, term, i, i + len(term)))
                    start = i + len(term)
            else:
                for m in re.finditer(rf"\b{re.escape(term)}\b", low):
                    hits.append((tech, term, m.start(), m.end()))
    return hits


def causal_hits(text):
    hits = []
    for name, pat in COMPILED_CAUSAL:
        for m in pat.finditer(text):
            hits.append((name, m.start(), m.end()))
    return hits


# --- Blame: no_sarcasm.py ----------------------------------------------------------------------------------------

CONTEXT_WINDOW = 300

CASE_INSENSITIVE_ENTITIES = {
    "figures": ["trump", "biden", "harris", "obama", "clinton", "pelosi", "mcconnell",
                "pence", "schumer", "sanders", "desantis"],
    "parties": ["democrats", "republicans", "liberals", "conservatives", "the left", "the right", "dems"],
    "institutions": ["government", "congress", "white house", "supreme court", "senate", "house"],
    "media": ["media", "mainstream media"],
    "groups": ["establishment", "elites", "deep state", "swamp"],
}
CASE_SENSITIVE_ENTITIES = {
    "parties": ["GOP", "MAGA", "DNC"],
    "institutions": ["FBI", "CIA", "DOJ"],
    "media": ["CNN", "FOX", "MSNBC", "NYT", "WAPO"],
    "organizations": ["UN", "WHO", "AOC"],
}
BLAME_PATTERNS = [
    r"\b(?:is|are|was|were)\s+(?:to\s+)?blame\b",
    r"\bresponsible\s+for\b",
    r"\b(?:it\'s|its|that\'s)\s+\w+\'s\s+fault\b",
    r"\bcaused\s+by\b",
    r"\bdue\s+to\b",
    r"\bbecause\s+of\b",
    r"\bblame\s+(?:it\s+)?on\b",
    r"\bscapegoat",
    r"\bthey\s+blame\b",
    r"\beverything\s+is\s+\w+\'s\s+fault\b",
    r"\bpointing\s+fingers?\b",
    r"\bblame\s+game\b",
]
DIRECT_PATTERNS = [
    (r"\b(?:is|are|was|were)\s+(?:directly\s+)?responsible\b", "direct responsibility"),
    (r"\b(?:is|are|was|were)\s+(?:to\s+)?blame\b", "direct blame"),
    (r"\bcaused\s+by\b", "causal attribution"),
    (r"\bdue\s+to\b", "due to"),
    (r"\bbecause\s+of\b", "because of"),
]
META_PATTERNS = [
    (r"\bthey\s+(?:always\s+)?blame\b", "they blame pattern"),
    (r"\bblame\s+(?:it\s+)?(?:all\s+)?on\b", "blame it on pattern"),
    (r"\bblame\s+everything\s+on\b", "blame everything pattern"),
    (r"\b(?:love|loves|quick|eager)\s+to\s+blame\b", "eager to blame"),
    (r"\bscapegoat", "scapegoating"),
    (r"\bfavorite\s+(?:target|scapegoat|thing\s+to\s+blame)\b", "favorite target"),
    (r"\bpointing\s+fingers?\b", "finger pointing"),
    (r"\bblame\s+game\b", "blame game"),
    (r"\balways\s+(?:blaming|blame)\b", "always blaming"),
    (r"\beverything\s+is\s+\w+\'s\s+fault\b", "everything is X's fault"),
    (r"\baccording\s+to\s+them\b", "according to them"),
    (r"\bthey\s+(?:say|claim|think)\s+(?:it\'s|its)\b", "they say it's"),
    (r"\b(?:you\s+know|I\s+mean|like)\s+they\s+(?:always\s+)?blame\b", "discourse marker + they blame"),
    (r"\b(?:so|well|anyway|gee)\s+(?:of\s+course|obviously)\s+it\'s\b", "discourse marker + attribution"),
    (r"\b(?:supposedly|allegedly|apparently)\s+it\'s\s+\w+\'s\s+fault\b", "hedged fault attribution"),
    (r"\b(?:quote\s+unquote|so-called)\s+responsible\b", "air quotes responsibility"),
    (r"\b(?:always|never|constantly)\s+(?:blaming|blame)\b", "emphatic blame frequency"),
    (r"\beverything\s+(?:has\s+to\s+be|is\s+always)\s+\w+\'s\s+fault\b", "everything is always X's fault"),
    (r"\bthe\s+(?:usual|typical|same\s+old)\s+blame\b", "typical blame pattern"),
    (r"\bhere\s+(?:we\s+go|they\s+go)\s+(?:again\s+)?(?:blaming|with\s+the\s+blame)\b", "here we go blaming"),
    (r"\b(?:oh|ah|gee)\s+(?:so\s+)?now\s+it\'s\s+\w+\'s\s+fault\b", "oh/gee now it's X's fault"),
    (r"\bthey\'re\s+(?:like|all)\s+[\"\']?(?:oh\s+)?it\'s\s+\w+\'s\s+fault\b", "reported blame speech"),
    (r"\beveryone\'s\s+(?:saying|going)\s+(?:oh\s+)?it\'s\b", "everyone's saying it's"),
    (r"\b(?:just|simply)\s+blame\s+(?:it\s+on|everything\s+on)\b", "just blame it on"),
    (r"\beasiest\s+(?:thing|way)\s+(?:is\s+)?(?:to\s+)?(?:just\s+)?blame\b", "easiest to blame"),
    (r"\bthey\s+(?:um|uh|like)\s+blame\b", "filler + blame"),
    (r"\bit\'s\s+(?:like|kinda|sorta)\s+\w+\'s\s+fault\b", "hedged fault"),
    (r"\bgee\s+.*\s+blame\b", "gee + blame pattern"),
]
DISCUSSION_MARKERS = [
    "who's to blame", "who is responsible", "finding someone to blame", "looking for someone to blame",
    "need someone to blame", "who else are they gonna blame", "what else is new they blame",
    "guess who gets the blame", "wonder who they'll blame",
]


def find_entities(text):
    found = []
    low = text.lower()
    for cat, ents in CASE_INSENSITIVE_ENTITIES.items():
        for e in ents:
            for m in re.finditer(rf"\b{re.escape(e)}\b", low, re.IGNORECASE):
                found.append({"entity": e, "category": cat, "start": m.start(), "end": m.end()})
    for cat, ents in CASE_SENSITIVE_ENTITIES.items():
        for e in ents:
            for m in re.finditer(rf"\b{re.escape(e)}\b", text):
                found.append({"entity": e, "category": cat, "start": m.start(), "end": m.end()})
    return found


def classify_blame(text):
    low = text.lower()
    direct = meta = 0
    reasons = []
    for p, d in DIRECT_PATTERNS:
        if re.search(p, low):
            direct += 3
            reasons.append(d)
    for p, d in META_PATTERNS:
        if re.search(p, low):  # as in no_sarcasm.py: case-sensitive on lower-cased text ("I mean" never fires)
            meta += 2
            reasons.append(d)
    if any(m in low for m in DISCUSSION_MARKERS):
        meta += 1
        reasons.append("discussing blame concept")
    if re.search(r"\b(?:if|when|would|could|might)\b.*\bblame\b", low):
        meta += 1
        reasons.append("hypothetical blame")
    total = direct + meta
    if total == 0:
        return "unclear", 0.5, reasons
    return ("meta" if meta > direct else "direct"), max(direct, meta) / total, reasons


def blame_instances(text):
    """no_sarcasm.py's process_file loop for one segment: one instance per (entity, blame match)."""
    out, seen = [], set()
    for ent in find_entities(text):
        lo = max(0, ent["start"] - CONTEXT_WINDOW)
        hi = min(len(text), ent["end"] + CONTEXT_WINDOW)
        ctx = text[lo:hi]
        for pat in BLAME_PATTERNS:
            for m in re.finditer(pat, ctx, re.IGNORECASE):
                key = (ent["entity"], m.group(), m.start())
                if key in seen:
                    continue
                seen.add(key)
                label, conf, reasons = classify_blame(ctx)
                out.append({"entity": ent["entity"], "category": ent["category"], "trigger": m.group(),
                            "trigger_start": lo + m.start(), "trigger_end": lo + m.end(),
                            "type": label, "confidence": round(conf, 2), "reasons": reasons})
    return out
