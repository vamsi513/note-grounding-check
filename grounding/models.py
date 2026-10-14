"""Data types shared by the loaders, the checkers and the evaluation script."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    number: int
    speaker: str
    text: str


@dataclass(frozen=True)
class Transcript:
    id: str
    visit_type: str
    turns: tuple[Turn, ...]

    def turn(self, number: int) -> Turn:
        for turn in self.turns:
            if turn.number == number:
                return turn
        raise KeyError(f"transcript {self.id} has no turn {number}")


@dataclass(frozen=True)
class Note:
    id: str
    transcript_id: str
    text: str


@dataclass
class SentenceVerdict:
    index: int
    text: str
    supported: bool
    evidence_turns: list[int] = field(default_factory=list)
    reason: str = ""


@dataclass
class NoteVerdict:
    note_id: str
    transcript_id: str
    checker: str
    sentences: list[SentenceVerdict]

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [s for s in self.sentences if not s.supported]


@dataclass(frozen=True)
class LabelledSentence:
    index: int
    text: str
    supported: bool
    evidence_turns: tuple[int, ...]
    error_type: str | None


@dataclass(frozen=True)
class NoteLabel:
    note_id: str
    transcript_id: str
    sentences: tuple[LabelledSentence, ...]
