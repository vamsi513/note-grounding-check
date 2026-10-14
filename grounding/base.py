"""The checker interface shared by the deterministic checker and the pluggable judges."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import NoteVerdict, SentenceVerdict, Transcript
from .sentences import split_sentences


@runtime_checkable
class SentenceChecker(Protocol):
    """Anything that can decide whether one note sentence is supported by a transcript."""

    name: str

    def check_sentence(
        self, index: int, sentence: str, transcript: Transcript
    ) -> SentenceVerdict: ...

    def check(self, note_text: str, transcript: Transcript, note_id: str = "") -> NoteVerdict: ...


class BaseChecker:
    """Splits a note into sentences and delegates each one to `check_sentence`."""

    name = "base"

    def check_sentence(self, index: int, sentence: str, transcript: Transcript) -> SentenceVerdict:
        raise NotImplementedError

    def check(self, note_text: str, transcript: Transcript, note_id: str = "") -> NoteVerdict:
        sentences = split_sentences(note_text)
        verdicts = [
            self.check_sentence(i, sentence, transcript) for i, sentence in enumerate(sentences)
        ]
        return NoteVerdict(note_id, transcript.id, self.name, verdicts)
