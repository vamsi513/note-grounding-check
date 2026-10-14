import json

import pytest

from grounding.datasets import (
    load_labels,
    load_notes,
    load_transcripts,
    parse_labels,
    parse_transcript,
)
from grounding.sentences import split_sentences


def test_the_shipped_dataset_has_thirty_visits():
    transcripts = load_transcripts()
    notes = load_notes()
    labels = load_labels()
    assert len(transcripts) == len(notes) == len(labels) == 30


def test_every_note_points_at_an_existing_transcript():
    transcripts = load_transcripts()
    for note in load_notes().values():
        assert note.transcript_id in transcripts


def test_transcript_lengths_are_within_the_documented_range():
    for transcript in load_transcripts().values():
        assert 4 <= len(transcript.turns) <= 12


def test_note_lengths_are_within_the_documented_range():
    for label in load_labels().values():
        assert 4 <= len(label.sentences) <= 8


def test_the_key_matches_what_the_splitter_produces():
    notes = load_notes()
    for note_id, label in load_labels().items():
        assert split_sentences(notes[note_id].text) == [s.text for s in label.sentences]


def test_every_evidence_turn_exists_in_its_transcript():
    transcripts = load_transcripts()
    for label in load_labels().values():
        count = len(transcripts[label.transcript_id].turns)
        for sentence in label.sentences:
            assert all(1 <= turn <= count for turn in sentence.evidence_turns)


def test_the_key_contains_the_documented_number_of_planted_errors():
    labels = load_labels()
    unsupported = [s for label in labels.values() for s in label.sentences if not s.supported]
    notes_with_errors = [
        label for label in labels.values() if any(not s.supported for s in label.sentences)
    ]
    assert len(unsupported) == 10
    assert len(notes_with_errors) == 10
    assert all(s.error_type for s in unsupported)


def test_all_four_error_kinds_are_present():
    labels = load_labels()
    kinds = {s.error_type for label in labels.values() for s in label.sentences if s.error_type}
    assert kinds == {"invented symptom", "invented dose", "wrong duration", "wrong medication"}


def test_a_supported_sentence_must_carry_evidence_turns():
    payload = {
        "n99": {
            "transcript_id": "t99",
            "sentences": [
                {
                    "index": 0,
                    "text": "A refill was sent.",
                    "supported": True,
                    "evidence_turns": [],
                    "error_type": None,
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="no evidence turn"):
        parse_labels(payload)


def test_out_of_order_sentence_indexes_are_rejected():
    payload = {
        "n99": {
            "transcript_id": "t99",
            "sentences": [
                {
                    "index": 3,
                    "text": "A refill was sent.",
                    "supported": False,
                    "evidence_turns": [],
                    "error_type": "invented dose",
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="out of order"):
        parse_labels(payload)


def test_turns_numbered_from_one_are_required():
    payload = {
        "id": "t99",
        "visit_type": "refill",
        "turns": [{"turn": 2, "speaker": "doctor", "text": "Hello."}],
    }
    with pytest.raises(ValueError, match="numbered from one"):
        parse_transcript(payload)


def test_the_loader_reads_from_a_given_path(tmp_path):
    payload = {
        "n01": {
            "transcript_id": "t01",
            "sentences": [
                {
                    "index": 0,
                    "text": "A refill was sent.",
                    "supported": True,
                    "evidence_turns": [2],
                    "error_type": None,
                }
            ],
        }
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    labels = load_labels(path)
    assert labels["n01"].sentences[0].evidence_turns == (2,)
