"""The deterministic grounding checker.

Each note sentence is matched against the transcript turns using lexical overlap, then three
hard rules are applied: a medication named in the note must appear in the conversation, every
quantity must match exactly, and the polarity of a shared term must agree.
"""

from __future__ import annotations

from .base import BaseChecker
from .evidence import (
    extract_quantities,
    is_question,
    negation_conflicts,
    polarity,
    quantity_satisfied,
)
from .models import SentenceVerdict, Transcript
from .text import content_tokens, is_number, medication_tokens, token_weight

DEFAULT_COVERAGE_THRESHOLD = 0.6
MAX_EVIDENCE_TURNS = 3


def _coverage(sentence_tokens: list[str], evidence_tokens: set[str]) -> float:
    total = sum(token_weight(t) for t in sentence_tokens)
    if total == 0:
        return 1.0
    hit = sum(token_weight(t) for t in sentence_tokens if t in evidence_tokens)
    return hit / total


def turn_token_sets(transcript: Transcript) -> dict[int, set[str]]:
    return {turn.number: set(content_tokens(turn.text)) for turn in transcript.turns}


def select_evidence_turns(
    sentence: str,
    sentence_tokens: list[str],
    transcript: Transcript,
    turn_tokens: dict[int, set[str]],
    max_turns: int,
) -> list[int]:
    """Greedily pick the turns that add the most weighted coverage, best first.

    Turns are ranked by the weight they newly cover, and ties are broken in favour of the turn
    whose polarity agrees with the sentence. Without that tie break a question such as
    "any dizziness?" would be picked ahead of the answer "no dizziness".
    """
    wanted = set(sentence_tokens)
    s_neg, s_aff = polarity(sentence)
    turn_polarity = {t.number: polarity(t.text) for t in transcript.turns}
    remaining = {n: tokens for n, tokens in turn_tokens.items() if tokens & wanted}
    chosen: list[int] = []
    covered: set[str] = set()

    def agreement(number: int, tokens: set[str]) -> int:
        t_neg, t_aff = turn_polarity[number]
        score = 0
        for token in tokens:
            sentence_negated = token in s_neg and token not in s_aff
            turn_negated = token in t_neg and token not in t_aff
            score += 1 if sentence_negated == turn_negated else -1
        return score

    while remaining and len(chosen) < max_turns:
        best_turn = None
        best_key = (0, 0)
        for number in sorted(remaining):
            new_tokens = {t for t in sentence_tokens if t in remaining[number] and t not in covered}
            gain = sum(token_weight(t) for t in new_tokens)
            if gain == 0:
                continue
            key = (gain, agreement(number, new_tokens))
            if best_turn is None or key > best_key:
                best_turn, best_key = number, key
        if best_turn is None:
            break
        chosen.append(best_turn)
        covered |= remaining.pop(best_turn) & wanted
    return sorted(chosen)


class DeterministicChecker(BaseChecker):
    """Lexical and numeric checker with no model and no network access."""

    name = "deterministic"

    def __init__(
        self,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
        max_evidence_turns: int = MAX_EVIDENCE_TURNS,
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.max_evidence_turns = max_evidence_turns

    def check_sentence(self, index: int, sentence: str, transcript: Transcript) -> SentenceVerdict:
        tokens = content_tokens(sentence)
        turn_tokens = turn_token_sets(transcript)
        evidence = select_evidence_turns(
            sentence, tokens, transcript, turn_tokens, self.max_evidence_turns
        )

        def union_tokens() -> set[str]:
            out: set[str] = set()
            for number in evidence:
                out |= turn_tokens[number]
            return out

        def union_text() -> str:
            return " . ".join(transcript.turn(n).text for n in evidence)

        def verdict(supported: bool, reason: str) -> SentenceVerdict:
            return SentenceVerdict(index, sentence, supported, sorted(evidence), reason)

        # Rule one: a medication named in the note must appear somewhere in the conversation.
        for med in medication_tokens(sentence):
            holders = [n for n, toks in turn_tokens.items() if med in toks]
            if not holders:
                return verdict(False, f"medication not in transcript: {med}")
            if not any(n in evidence for n in holders):
                evidence.append(holders[0])
                evidence.sort()

        # Rule two: every quantity in the note must match a quantity in the evidence exactly.
        for quantity in extract_quantities(sentence):
            if quantity_satisfied(quantity, extract_quantities(union_text())):
                continue
            added = False
            for number in sorted(turn_tokens):
                turn = transcript.turn(number)
                shares_term = bool({t for t in tokens if not is_number(t)} & turn_tokens[number])
                if shares_term and quantity_satisfied(quantity, extract_quantities(turn.text)):
                    evidence.append(number)
                    evidence.sort()
                    added = True
                    break
            if not added:
                return verdict(False, f"quantity not supported: {quantity}")

        # Rule three: polarity of a shared term must agree with every turn that asserts it.
        # Question turns are skipped because raising a symptom is not the same as affirming it.
        asserting = " . ".join(
            turn.text
            for turn in transcript.turns
            if turn_tokens[turn.number] & set(tokens) and not is_question(turn.text)
        )
        conflicts = negation_conflicts(sentence, asserting)
        if conflicts:
            return verdict(False, "negation mismatch on: " + ", ".join(conflicts))

        coverage = _coverage(tokens, union_tokens())
        if coverage < self.coverage_threshold:
            return verdict(False, f"insufficient lexical support (coverage {coverage:.2f})")
        return verdict(True, f"lexical coverage {coverage:.2f}")
