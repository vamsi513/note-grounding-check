from grounding.checker import DeterministicChecker
from grounding.evidence import is_question, negation_conflicts, polarity


def test_denies_is_supported_by_no():
    assert negation_conflicts("Patient denies chest pain.", "No chest pain.") == []


def test_denies_is_not_supported_by_has():
    assert negation_conflicts("Patient denies chest pain.", "I have chest pain today.") == [
        "chest",
        "pain",
    ]


def test_an_affirmed_claim_conflicts_with_a_denial():
    assert negation_conflicts("Patient reports chest pain.", "No chest pain at all.") == [
        "chest",
        "pain",
    ]


def test_negation_scope_stops_at_a_contrastive_conjunction():
    negated, affirmed = polarity("Some fatigue, but no cold intolerance.")
    assert "fatigue" in affirmed
    assert "cold" in negated
    assert "fatigue" not in negated


def test_negation_scope_stops_at_a_full_stop():
    negated, affirmed = polarity("No chest pain. I do get short of breath on stairs.")
    assert "chest" in negated
    assert "breath" in affirmed


def test_a_term_asserted_both_ways_is_not_a_conflict():
    evidence = "No fever today . Call if you get a fever next week"
    assert negation_conflicts("Patient denies fever.", evidence) == []


def test_questions_are_recognised():
    assert is_question("Any chest pain?")
    assert not is_question("No chest pain.")


def test_denial_is_supported_by_the_answer_not_the_question(refill_transcript):
    checker = DeterministicChecker()
    verdict = checker.check_sentence(0, "Patient denies chest pain.", refill_transcript)
    assert verdict.supported
    assert verdict.evidence_turns == [6]


def test_claiming_a_denied_symptom_is_unsupported(refill_transcript):
    checker = DeterministicChecker()
    verdict = checker.check_sentence(0, "Patient reports chest pain.", refill_transcript)
    assert not verdict.supported
    assert "negation mismatch" in verdict.reason
