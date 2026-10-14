"""Quantity extraction and negation scope used by the deterministic checker."""

from __future__ import annotations

from dataclasses import dataclass

from .text import (
    CLAUSE_SPLIT_RE,
    FRAMING,
    NEGATION_CUES,
    STOPWORDS,
    UNITS,
    is_number,
    tokenize,
)


@dataclass(frozen=True)
class Quantity:
    """A number from a note or a turn, with the unit that follows it when there is one."""

    value: str
    unit: str | None

    def __str__(self) -> str:
        return f"{self.value} {self.unit}" if self.unit else self.value


MAX_UNIT_LOOKAHEAD = 2


def _unit_after(tokens: list[str], index: int) -> str | None:
    """Find the unit that belongs to the number at `index`.

    A short run of filler words is stepped over so that "eight more weeks" binds weeks to
    eight. The search stops at the next number so that "134 over 86" yields two bare values.
    """
    position = index + 1
    skipped = 0
    while position < len(tokens) and skipped <= MAX_UNIT_LOOKAHEAD:
        token = tokens[position]
        if token in UNITS:
            return token
        if is_number(token):
            return None
        if token in STOPWORDS:
            position += 1
            skipped += 1
            continue
        return None
    return None


def extract_quantities(text: str) -> list[Quantity]:
    """Pull every number out of a piece of text, binding its unit when there is one."""
    tokens = tokenize(text)
    return [
        Quantity(token, _unit_after(tokens, index))
        for index, token in enumerate(tokens)
        if is_number(token)
    ]


def quantity_satisfied(needle: Quantity, haystack: list[Quantity]) -> bool:
    """A quantity is matched when the value matches and the units do not conflict."""
    for candidate in haystack:
        if candidate.value != needle.value:
            continue
        if needle.unit is None or candidate.unit is None:
            return True
        if needle.unit == candidate.unit:
            return True
    return False


def is_question(text: str) -> bool:
    """Questions raise a term without asserting it, so they carry no polarity evidence."""
    return text.strip().endswith("?")


def polarity(text: str) -> tuple[set[str], set[str]]:
    """Split the content tokens of a text into negated and affirmed sets.

    Negation scope runs from a cue to the end of its clause, where clauses are bounded by
    punctuation and by contrastive conjunctions.
    """
    negated: set[str] = set()
    affirmed: set[str] = set()
    for clause in CLAUSE_SPLIT_RE.split(text.lower()):
        tokens = tokenize(clause)
        in_scope = False
        for token in tokens:
            if token in NEGATION_CUES:
                in_scope = True
                continue
            if token in STOPWORDS or token in FRAMING:
                continue
            (negated if in_scope else affirmed).add(token)
    return negated, affirmed


def negation_conflicts(sentence: str, evidence: str) -> list[str]:
    """Tokens whose polarity in the sentence contradicts their polarity in the evidence."""
    s_neg, s_aff = polarity(sentence)
    e_neg, e_aff = polarity(evidence)
    conflicts: list[str] = []
    for token in sorted(s_neg):
        if token in e_aff and token not in e_neg:
            conflicts.append(token)
    for token in sorted(s_aff):
        if token in e_neg and token not in e_aff:
            conflicts.append(token)
    return conflicts
