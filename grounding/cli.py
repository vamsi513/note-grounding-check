"""Command line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .checker import DeterministicChecker
from .datasets import (
    LABELS_PATH,
    NOTE_DIR,
    TRANSCRIPT_DIR,
    load_labels,
    load_note,
    load_notes,
    load_transcript,
    load_transcripts,
)
from .evaluate import dataset_summary, evaluate_checker, format_report
from .judges import OFFLINE_CHECKERS, build_checker


def _read_note_text(path: Path) -> str:
    if path.suffix == ".json":
        return load_note(path).text
    return path.read_text(encoding="utf-8")


def run_check(args: argparse.Namespace, stream=None) -> int:
    stream = sys.stdout if stream is None else stream
    transcript = load_transcript(args.transcript)
    note_text = _read_note_text(Path(args.note))
    checker = build_checker(args.checker)
    result = checker.check(note_text, transcript, Path(args.note).stem)

    print(f"transcript: {transcript.id} ({transcript.visit_type})", file=stream)
    print(f"checker:    {checker.name}", file=stream)
    print("", file=stream)
    for verdict in result.sentences:
        status = "supported" if verdict.supported else "UNSUPPORTED"
        turns = ", ".join(str(t) for t in verdict.evidence_turns) or "none"
        print(f"[{verdict.index}] {status}", file=stream)
        print(f"    sentence: {verdict.text}", file=stream)
        print(f"    evidence turns: {turns}", file=stream)
        print(f"    reason: {verdict.reason}", file=stream)
    flagged = len(result.unsupported)
    print("", file=stream)
    print(f"{flagged} of {len(result.sentences)} sentences unsupported", file=stream)
    return 1 if flagged else 0


def run_evaluate(args: argparse.Namespace, stream=None) -> int:
    stream = sys.stdout if stream is None else stream
    notes = load_notes(args.notes)
    transcripts = load_transcripts(args.transcripts)
    labels = load_labels(args.labels)
    results = [
        evaluate_checker(build_checker(name), notes, transcripts, labels)
        for name in sorted(OFFLINE_CHECKERS)
    ]
    report = format_report(dataset_summary(labels), results)
    print(report, file=stream, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nwritten to {out}", file=stream)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m grounding",
        description=(
            "Check whether each sentence of a generated clinical style note is supported by "
            "the source conversation. A verification study, not clinical software."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check one note against one transcript")
    check.add_argument("--transcript", required=True, help="path to a transcript JSON file")
    check.add_argument("--note", required=True, help="path to a note JSON file or a text file")
    check.add_argument(
        "--checker",
        default=DeterministicChecker.name,
        help="checker to use (default: deterministic)",
    )
    check.set_defaults(handler=run_check)

    evaluate = sub.add_parser("evaluate", help="score the offline checkers against the key")
    evaluate.add_argument("--transcripts", default=str(TRANSCRIPT_DIR))
    evaluate.add_argument("--notes", default=str(NOTE_DIR))
    evaluate.add_argument("--labels", default=str(LABELS_PATH))
    evaluate.add_argument("--out", default=None, help="also write the report to this path")
    evaluate.set_defaults(handler=run_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
