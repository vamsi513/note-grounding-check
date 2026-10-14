import pytest

from grounding.models import Transcript, Turn

REFILL_TURNS = [
    ("doctor", "What brings you in today?"),
    ("patient", "I need a refill of my lisinopril. I have about a week left."),
    ("doctor", "You are on lisinopril 10 mg once daily, correct?"),
    ("patient", "Yes, 10 mg every morning."),
    ("doctor", "Any dizziness or chest pain since we started it?"),
    ("patient", "No dizziness and no chest pain."),
    ("doctor", "I will send a 90 day supply to your pharmacy."),
]


@pytest.fixture
def refill_transcript() -> Transcript:
    turns = tuple(
        Turn(number=i, speaker=speaker, text=text)
        for i, (speaker, text) in enumerate(REFILL_TURNS, start=1)
    )
    return Transcript(id="tx", visit_type="refill", turns=turns)
