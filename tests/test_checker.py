from grounding.checker import DeterministicChecker, select_evidence_turns, turn_token_sets
from grounding.text import content_tokens


def test_evidence_turns_point_at_the_dose_turn(refill_transcript):
    verdict = DeterministicChecker().check_sentence(
        0, "Dose is lisinopril 10 mg once daily.", refill_transcript
    )
    assert verdict.supported
    assert 3 in verdict.evidence_turns


def test_evidence_turns_are_sorted_and_within_the_transcript(refill_transcript):
    verdict = DeterministicChecker().check_sentence(
        0, "A 90 day supply was sent to the pharmacy.", refill_transcript
    )
    assert verdict.evidence_turns == sorted(verdict.evidence_turns)
    assert all(1 <= t <= len(refill_transcript.turns) for t in verdict.evidence_turns)


def test_evidence_selection_is_capped(refill_transcript):
    tokens = content_tokens("lisinopril 10 mg daily refill dizziness chest pain supply pharmacy")
    chosen = select_evidence_turns(
        "lisinopril 10 mg daily refill dizziness chest pain supply pharmacy",
        tokens,
        refill_transcript,
        turn_token_sets(refill_transcript),
        2,
    )
    assert len(chosen) <= 2


def test_a_medication_not_in_the_conversation_is_unsupported(refill_transcript):
    verdict = DeterministicChecker().check_sentence(
        0, "Patient takes atenolol 10 mg once daily.", refill_transcript
    )
    assert not verdict.supported
    assert "atenolol" in verdict.reason


def test_a_medication_in_the_conversation_is_accepted(refill_transcript):
    verdict = DeterministicChecker().check_sentence(
        0, "Patient requests a refill of lisinopril.", refill_transcript
    )
    assert verdict.supported


def test_an_invented_symptom_has_no_lexical_support(refill_transcript):
    verdict = DeterministicChecker().check_sentence(
        0, "Patient also reports a mild rash.", refill_transcript
    )
    assert not verdict.supported
    assert verdict.evidence_turns == []


def test_checking_a_note_returns_one_verdict_per_sentence(refill_transcript):
    note = "Patient denies dizziness. Patient denies chest pain. Patient takes atenolol."
    result = DeterministicChecker().check(note, refill_transcript, "n-test")
    assert [v.index for v in result.sentences] == [0, 1, 2]
    assert [v.supported for v in result.sentences] == [True, True, False]
    assert len(result.unsupported) == 1
    assert result.note_id == "n-test"
    assert result.checker == "deterministic"


def test_a_raised_coverage_threshold_rejects_a_loose_paraphrase(refill_transcript):
    sentence = "Patient wants the lisinopril renewed."
    assert (
        DeterministicChecker(coverage_threshold=0.1)
        .check_sentence(0, sentence, refill_transcript)
        .supported
    )
    assert (
        not DeterministicChecker(coverage_threshold=0.9)
        .check_sentence(0, sentence, refill_transcript)
        .supported
    )
