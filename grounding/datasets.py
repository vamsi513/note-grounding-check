"""Loaders for the synthetic transcripts, the notes and the answer key."""

from __future__ import annotations

import json
from pathlib import Path

from .models import LabelledSentence, Note, NoteLabel, Transcript, Turn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
NOTE_DIR = DATA_DIR / "notes"
LABELS_PATH = DATA_DIR / "labels.json"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_transcript(payload: dict) -> Transcript:
    turns = tuple(
        Turn(number=int(t["turn"]), speaker=str(t["speaker"]), text=str(t["text"]))
        for t in payload["turns"]
    )
    if not turns:
        raise ValueError(f"transcript {payload.get('id')} has no turns")
    expected = tuple(range(1, len(turns) + 1))
    if tuple(t.number for t in turns) != expected:
        raise ValueError(f"transcript {payload.get('id')} turns are not numbered from one")
    return Transcript(id=str(payload["id"]), visit_type=str(payload["visit_type"]), turns=turns)


def load_transcript(path: str | Path) -> Transcript:
    return parse_transcript(_read_json(Path(path)))


def load_note(path: str | Path) -> Note:
    payload = _read_json(Path(path))
    return Note(
        id=str(payload["id"]),
        transcript_id=str(payload["transcript_id"]),
        text=str(payload["text"]),
    )


def load_transcripts(directory: str | Path = TRANSCRIPT_DIR) -> dict[str, Transcript]:
    out: dict[str, Transcript] = {}
    for path in sorted(Path(directory).glob("*.json")):
        transcript = load_transcript(path)
        out[transcript.id] = transcript
    return out


def load_notes(directory: str | Path = NOTE_DIR) -> dict[str, Note]:
    out: dict[str, Note] = {}
    for path in sorted(Path(directory).glob("*.json")):
        note = load_note(path)
        out[note.id] = note
    return out


def parse_labels(payload: dict) -> dict[str, NoteLabel]:
    out: dict[str, NoteLabel] = {}
    for note_id, entry in payload.items():
        sentences = []
        for position, item in enumerate(entry["sentences"]):
            index = int(item["index"])
            if index != position:
                raise ValueError(f"{note_id} sentence indexes are out of order at {position}")
            supported = item["supported"]
            if not isinstance(supported, bool):
                raise ValueError(f"{note_id} sentence {index} has a non boolean label")
            turns = tuple(int(t) for t in item["evidence_turns"])
            if supported and not turns:
                raise ValueError(f"{note_id} sentence {index} is supported with no evidence turn")
            sentences.append(
                LabelledSentence(
                    index=index,
                    text=str(item["text"]),
                    supported=supported,
                    evidence_turns=turns,
                    error_type=item["error_type"],
                )
            )
        out[note_id] = NoteLabel(
            note_id=note_id,
            transcript_id=str(entry["transcript_id"]),
            sentences=tuple(sentences),
        )
    return out


def load_labels(path: str | Path = LABELS_PATH) -> dict[str, NoteLabel]:
    return parse_labels(_read_json(Path(path)))
