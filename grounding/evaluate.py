"""Evaluate the offline checkers against the answer key in data/labels.json.

The positive class is "unsupported", because the job of the tool is to flag claims that the
conversation does not support. So a false positive is a supported sentence that was flagged,
and a false negative is an unsupported sentence that was missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseChecker
from .models import Note, NoteLabel, Transcript


@dataclass
class Miss:
    kind: str
    note_id: str
    sentence_index: int
    text: str
    detail: str


@dataclass
class Metrics:
    checker: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    evidence_hits: int = 0
    evidence_checked: int = 0
    misses: list[Miss] = field(default_factory=list)
    by_error_type: dict[str, list[int]] = field(default_factory=dict)

    @property
    def sentences(self) -> int:
        return (
            self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        )

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def evidence_turn_rate(self) -> float:
        return self.evidence_hits / self.evidence_checked if self.evidence_checked else 0.0


def evaluate_checker(
    checker: BaseChecker,
    notes: dict[str, Note],
    transcripts: dict[str, Transcript],
    labels: dict[str, NoteLabel],
) -> Metrics:
    metrics = Metrics(checker=checker.name)
    for note_id in sorted(labels):
        label = labels[note_id]
        note = notes[note_id]
        transcript = transcripts[label.transcript_id]
        result = checker.check(note.text, transcript, note_id)
        if len(result.sentences) != len(label.sentences):
            raise ValueError(
                f"{note_id}: checker produced {len(result.sentences)} sentences but the key "
                f"has {len(label.sentences)}"
            )
        for verdict, truth in zip(result.sentences, label.sentences, strict=True):
            if truth.error_type:
                counts = metrics.by_error_type.setdefault(truth.error_type, [0, 0])
                counts[1] += 1
            if not verdict.supported and not truth.supported:
                metrics.true_positives += 1
                if truth.error_type:
                    metrics.by_error_type[truth.error_type][0] += 1
            elif not verdict.supported and truth.supported:
                metrics.false_positives += 1
                metrics.misses.append(
                    Miss("false positive", note_id, truth.index, truth.text, verdict.reason)
                )
            elif verdict.supported and not truth.supported:
                metrics.false_negatives += 1
                metrics.misses.append(
                    Miss(
                        "false negative",
                        note_id,
                        truth.index,
                        truth.text,
                        truth.error_type or "unsupported",
                    )
                )
            else:
                metrics.true_negatives += 1
            if truth.supported and verdict.supported:
                metrics.evidence_checked += 1
                if set(verdict.evidence_turns) & set(truth.evidence_turns):
                    metrics.evidence_hits += 1
    return metrics


def dataset_summary(labels: dict[str, NoteLabel]) -> dict[str, int]:
    sentences = [s for label in labels.values() for s in label.sentences]
    return {
        "notes": len(labels),
        "sentences": len(sentences),
        "unsupported sentences": sum(1 for s in sentences if not s.supported),
        "notes with at least one unsupported sentence": sum(
            1 for label in labels.values() if any(not s.supported for s in label.sentences)
        ),
    }


def format_report(summary: dict[str, int], results: list[Metrics]) -> str:
    lines: list[str] = []
    lines.append("# Measured results")
    lines.append("")
    lines.append(
        "Produced by `python -m grounding evaluate`. The positive class is a sentence the "
        "conversation does not support, so a false positive is a supported sentence that was "
        "flagged and a false negative is an unsupported sentence that was missed."
    )
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("| Quantity | Count |")
    lines.append("| --- | --- |")
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## Checker results")
    lines.append("")
    lines.append(
        "| Checker | Sentences | Precision | Recall | F1 | True positives | "
        "False positives | False negatives | True negatives |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for metrics in results:
        lines.append(
            f"| {metrics.checker} | {metrics.sentences} | {metrics.precision:.3f} | "
            f"{metrics.recall:.3f} | {metrics.f1:.3f} | {metrics.true_positives} | "
            f"{metrics.false_positives} | {metrics.false_negatives} | "
            f"{metrics.true_negatives} |"
        )
    lines.append("")
    lines.append("## Evidence turns")
    lines.append("")
    lines.append(
        "Share of correctly supported sentences where the reported evidence turns overlap the "
        "turns recorded in the answer key."
    )
    lines.append("")
    lines.append("| Checker | Sentences compared | Overlapping | Rate |")
    lines.append("| --- | --- | --- | --- |")
    for metrics in results:
        lines.append(
            f"| {metrics.checker} | {metrics.evidence_checked} | {metrics.evidence_hits} | "
            f"{metrics.evidence_turn_rate:.3f} |"
        )
    lines.append("")
    lines.append("## Detection by error type")
    lines.append("")
    error_types = sorted({k for m in results for k in m.by_error_type})
    header = "| Error type | " + " | ".join(m.checker for m in results) + " |"
    lines.append(header)
    lines.append("| --- |" + " --- |" * len(results))
    for error_type in error_types:
        cells = []
        for metrics in results:
            found, total = metrics.by_error_type.get(error_type, [0, 0])
            cells.append(f"{found} of {total}")
        lines.append(f"| {error_type} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Individual misses")
    lines.append("")
    any_miss = False
    for metrics in results:
        if not metrics.misses:
            continue
        any_miss = True
        lines.append(f"### {metrics.checker}")
        lines.append("")
        lines.append("| Kind | Note | Sentence | Text | Detail |")
        lines.append("| --- | --- | --- | --- | --- |")
        for miss in metrics.misses:
            lines.append(
                f"| {miss.kind} | {miss.note_id} | {miss.sentence_index} | {miss.text} | "
                f"{miss.detail} |"
            )
        lines.append("")
    if not any_miss:
        lines.append("No misses.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
