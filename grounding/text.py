"""Tokenisation, light stemming and vocabulary used by the checkers."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
NUMBER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
NUMERIC_PLURAL_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)s$")

# Suffix, smallest stem length that may remain, replacement for the suffix.
SUFFIX_RULES: tuple[tuple[str, int, str], ...] = (
    ("ness", 4, ""),
    ("ing", 4, ""),
    ("ies", 3, "y"),
    ("ed", 4, ""),
    ("s", 3, ""),
)

# Number words that must be compared as quantities, not as bare vocabulary.
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "once": "1",
    "two": "2",
    "twice": "2",
    "three": "3",
    "thrice": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def _stem(token: str) -> str:
    """Strip a few common English suffixes so spoken and written forms line up."""
    if NUMBER_RE.match(token):
        return token
    plural = NUMERIC_PLURAL_RE.match(token)
    if plural:
        return plural.group(1)
    for suffix, min_stem, replacement in SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)] + replacement
    return token


def _with_stems(words: list[str]) -> frozenset[str]:
    """A vocabulary set that matches a word before and after stemming."""
    return frozenset(words) | frozenset(_stem(w) for w in words)


# Words that carry no evidence value in either a note or a transcript turn.
STOPWORDS = _with_stems(
    """
    a about again all am an and another any are as at back be been before being both but by
    can could did do does doing done for from get gets getting got had has have he her here
    him his how i if in into is it its just let lets like me more most much my no none not
    now of on once one only or other our out over own she should since so some still such
    than that the their them then there these they this those to too up us very was we were
    what when where which while who will with would yes you your
    """.split()
)

# Framing language that appears in written notes but never in spoken turns. Removing it keeps
# lexical coverage comparable between the two sides.
FRAMING = _with_stems(
    """
    patient pt doctor clinician provider report reports reported reporting state states stated
    deny denies denied note noted noting advise advised advises asked ask asking current
    currently examination exam finding findings plan plans today visit also additionally
    overall discussed discussion
    """.split()
)

# Units and frequency words that bind to a preceding quantity.
UNITS = _with_stems(
    """
    mg mcg g gram kg lb pound ml unit tablet capsule puff dose
    day week month year hour minute
    percent session refill reading
    daily nightly weekly monthly time
    """.split()
)

# Generic medication names used in the synthetic data plus common near neighbours, so that a
# swapped or invented drug name is a hard mismatch rather than a lexical near miss.
MEDICATIONS = _with_stems(
    """
    acetaminophen albuterol amitriptyline amlodipine amoxicillin aspirin atenolol atorvastatin
    azithromycin carvedilol cephalexin cetirizine citalopram clopidogrel doxycycline duloxetine
    escitalopram ferrous fluoxetine fluticasone furosemide gabapentin glipizide
    hydrochlorothiazide ibuprofen insulin levothyroxine lisinopril loratadine losartan
    meloxicam metformin metoprolol metronidazole montelukast naproxen omeprazole pantoprazole
    prednisone pravastatin propranolol rosuvastatin sertraline simvastatin sumatriptan
    tamsulosin tramadol trazodone valsartan warfarin
    """.split()
)

NEGATION_CUES = _with_stems(
    """
    no not none never without denies denied deny negative absent lacks lacking
    """.split()
)

CLAUSE_SPLIT_RE = re.compile(r"[.,;:!?]+|\bbut\b|\bhowever\b|\balthough\b")


def tokenize(text: str) -> list[str]:
    """Lowercase, split and stem a piece of text, mapping number words to digits."""
    out: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = NUMBER_WORDS.get(raw, raw)
        stemmed = _stem(token)
        out.append(NUMBER_WORDS.get(stemmed, stemmed))
    return out


def is_number(token: str) -> bool:
    return bool(NUMBER_RE.match(token))


def is_evidence_bearing(token: str) -> bool:
    return token not in STOPWORDS and token not in FRAMING


def content_tokens(text: str) -> list[str]:
    """Tokens that can serve as evidence, in order, with duplicates removed."""
    seen: set[str] = set()
    out: list[str] = []
    for token in tokenize(text):
        if not is_evidence_bearing(token) or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def token_weight(token: str) -> int:
    """Medication names and quantities carry more evidence weight than ordinary words."""
    return 3 if is_number(token) or token in MEDICATIONS else 1


def medication_tokens(text: str) -> list[str]:
    return [t for t in content_tokens(text) if t in MEDICATIONS]
