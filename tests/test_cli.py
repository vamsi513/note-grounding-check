import io

import pytest

from grounding.cli import main
from grounding.datasets import NOTE_DIR, TRANSCRIPT_DIR


def transcript(name: str) -> str:
    return str(TRANSCRIPT_DIR / name)


def note(name: str) -> str:
    return str(NOTE_DIR / name)


def test_check_reports_each_sentence_and_its_evidence_turns(capsys):
    code = main(
        [
            "check",
            "--transcript",
            transcript("t01.json"),
            "--note",
            note("n01.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "transcript: t01" in out
    assert "checker:    deterministic" in out
    assert "evidence turns:" in out
    assert "0 of 5 sentences unsupported" in out


def test_check_exits_non_zero_when_a_claim_is_unsupported(capsys):
    code = main(
        [
            "check",
            "--transcript",
            transcript("t08.json"),
            "--note",
            note("n08.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "UNSUPPORTED" in out
    assert "quantity not supported: 100 mcg" in out


def test_check_accepts_a_plain_text_note(tmp_path, capsys):
    text_note = tmp_path / "note.txt"
    text_note.write_text("Patient denies dizziness.\n", encoding="utf-8")
    code = main(["check", "--transcript", transcript("t01.json"), "--note", str(text_note)])
    assert code == 0
    assert "supported" in capsys.readouterr().out


def test_check_can_use_the_mock_judge(capsys):
    code = main(
        [
            "check",
            "--transcript",
            transcript("t08.json"),
            "--note",
            note("n08.json"),
            "--checker",
            "mock-judge",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "checker:    mock-judge" in out


def test_evaluate_prints_a_report_for_both_offline_checkers(capsys):
    assert main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "# Measured results" in out
    assert "deterministic" in out
    assert "mock-judge" in out


def test_evaluate_writes_the_report_to_a_path(tmp_path):
    out_path = tmp_path / "nested" / "RESULTS.md"
    stream = io.StringIO()
    from grounding.cli import build_parser

    args = build_parser().parse_args(["evaluate", "--out", str(out_path)])
    assert args.handler(args, stream=stream) == 0
    assert out_path.read_text(encoding="utf-8").startswith("# Measured results")


def test_an_unknown_checker_is_rejected():
    with pytest.raises(KeyError):
        main(
            [
                "check",
                "--transcript",
                transcript("t01.json"),
                "--note",
                note("n01.json"),
                "--checker",
                "nope",
            ]
        )
