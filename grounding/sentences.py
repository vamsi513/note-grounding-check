"""Sentence splitting for short clinical style notes."""

from __future__ import annotations

import re

_DECIMAL_MARK = "\x00"
_ABBREV_MARK = "\x01"

# Abbreviations whose trailing full stop does not end a sentence.
ABBREVIATIONS = (
    "dr",
    "mr",
    "mrs",
    "ms",
    "approx",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
)

_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def _protect(text: str) -> str:
    text = _DECIMAL_RE.sub(_DECIMAL_MARK, text)
    for abbrev in ABBREVIATIONS:
        pattern = re.compile(rf"\b{re.escape(abbrev)}\.", re.IGNORECASE)
        text = pattern.sub(lambda m: m.group(0)[:-1] + _ABBREV_MARK, text)
    return text


def _restore(text: str) -> str:
    return text.replace(_DECIMAL_MARK, ".").replace(_ABBREV_MARK, ".")


def split_sentences(text: str) -> list[str]:
    """Split a note into sentences.

    Decimal points and a small set of abbreviations are protected so that doses such as
    2.5 mg do not create spurious sentence boundaries.
    """
    if not text or not text.strip():
        return []
    protected = _protect(text)
    parts = _SPLIT_RE.split(protected)
    out: list[str] = []
    for part in parts:
        sentence = _WS_RE.sub(" ", _restore(part)).strip()
        if sentence:
            out.append(sentence)
    return out
