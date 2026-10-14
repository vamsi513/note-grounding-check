import pytest

from grounding.checker import DeterministicChecker
from grounding.datasets import load_labels, load_notes, load_transcripts
from grounding.evaluate import Metrics, dataset_summary, evaluate_checker, format_report
from grounding.judges import MockLexicalJudge, ScriptedJudge


@pytest.fixture(scope="module")
def dataset():
    return load_notes(), load_transcripts(), load_labels()


def test_metrics_arithmetic():
    metrics = Metrics(
        "x", true_positives=8, false_positives=2, false_negatives=2, true_negatives=88
    )
    assert metrics.sentences == 100
    assert metrics.precision == pytest.approx(0.8)
    assert metrics.recall == pytest.approx(0.8)
    assert metrics.f1 == pytest.approx(0.8)


def test_metrics_are_zero_when_nothing_is_flagged():
    metrics = Metrics("x", true_negatives=10)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.evidence_turn_rate == 0.0


def test_dataset_summary_counts_the_shipped_data(dataset):
    _, _, labels = dataset
    summary = dataset_summary(labels)
    assert summary["notes"] == 30
    assert summary["unsupported sentences"] == 10
    assert summary["notes with at least one unsupported sentence"] == 10
    assert summary["sentences"] == sum(len(label.sentences) for label in labels.values())


def test_every_sentence_is_counted_exactly_once(dataset):
    notes, transcripts, labels = dataset
    metrics = evaluate_checker(DeterministicChecker(), notes, transcripts, labels)
    assert metrics.sentences == sum(len(label.sentences) for label in labels.values())


def test_a_perfect_judge_scores_one(dataset):
    notes, transcripts, labels = dataset
    answers = {
        sentence.text: sentence.supported
        for label in labels.values()
        for sentence in label.sentences
    }
    metrics = evaluate_checker(ScriptedJudge(answers), notes, transcripts, labels)
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_a_judge_that_flags_nothing_has_no_recall(dataset):
    notes, transcripts, labels = dataset
    metrics = evaluate_checker(ScriptedJudge({}, default=True), notes, transcripts, labels)
    assert metrics.true_positives == 0
    assert metrics.false_negatives == 10
    assert metrics.recall == 0.0


def test_misses_are_recorded_with_their_note_and_sentence(dataset):
    notes, transcripts, labels = dataset
    metrics = evaluate_checker(MockLexicalJudge(), notes, transcripts, labels)
    assert metrics.false_negatives == len([m for m in metrics.misses if m.kind == "false negative"])
    for miss in metrics.misses:
        assert miss.note_id in labels
        assert 0 <= miss.sentence_index < len(labels[miss.note_id].sentences)


def test_error_type_tallies_cover_every_planted_error(dataset):
    notes, transcripts, labels = dataset
    metrics = evaluate_checker(DeterministicChecker(), notes, transcripts, labels)
    assert sum(total for _, total in metrics.by_error_type.values()) == 10
    for found, total in metrics.by_error_type.values():
        assert 0 <= found <= total


def test_the_report_contains_the_tables_and_the_numbers(dataset):
    notes, transcripts, labels = dataset
    results = [
        evaluate_checker(DeterministicChecker(), notes, transcripts, labels),
        evaluate_checker(MockLexicalJudge(), notes, transcripts, labels),
    ]
    report = format_report(dataset_summary(labels), results)
    assert "# Measured results" in report
    assert "## Checker results" in report
    assert "## Detection by error type" in report
    assert "deterministic" in report
    assert "mock-judge" in report
    assert f"| {results[0].true_positives} |" in report


def test_a_mismatched_key_length_is_an_error(dataset):
    notes, transcripts, labels = dataset
    note_id = "n01"
    trimmed = {note_id: labels[note_id]}
    shortened = type(labels[note_id])(
        note_id=note_id,
        transcript_id=labels[note_id].transcript_id,
        sentences=labels[note_id].sentences[:-1],
    )
    trimmed[note_id] = shortened
    with pytest.raises(ValueError, match="sentences but the key"):
        evaluate_checker(DeterministicChecker(), notes, transcripts, trimmed)
