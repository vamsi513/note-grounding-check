"""Pluggable judges.

The mock judge is a lexical overlap baseline. It exists so that the judge interface is
exercised by the test suite and so that the evaluation has a weaker reference point to compare
the deterministic checker against. It is not a language model.

The optional language model judge is off by default, is not part of the registry used by the
evaluation, and raises before it would ever reach the network when it has no configuration.
It is never called by the tests or by continuous integration.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .base import BaseChecker
from .checker import DeterministicChecker, select_evidence_turns, turn_token_sets
from .models import SentenceVerdict, Transcript
from .text import content_tokens, token_weight

DEFAULT_COVERAGE_THRESHOLD = 0.6
URL_ENV = "GROUNDING_JUDGE_URL"
KEY_ENV = "GROUNDING_JUDGE_API_KEY"
MODEL_ENV = "GROUNDING_JUDGE_MODEL"


class JudgeNotConfigured(RuntimeError):
    """Raised when the optional judge is asked to run without being configured."""


class MockLexicalJudge(BaseChecker):
    """Overlap only baseline: no quantity rule, no medication rule, no negation rule."""

    name = "mock-judge"

    def __init__(self, coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> None:
        self.coverage_threshold = coverage_threshold

    def check_sentence(self, index: int, sentence: str, transcript: Transcript) -> SentenceVerdict:
        tokens = content_tokens(sentence)
        turn_tokens = turn_token_sets(transcript)
        evidence = select_evidence_turns(sentence, tokens, transcript, turn_tokens, 3)
        union: set[str] = set()
        for number in evidence:
            union |= turn_tokens[number]
        total = sum(token_weight(t) for t in tokens)
        hit = sum(token_weight(t) for t in tokens if t in union)
        coverage = 1.0 if total == 0 else hit / total
        supported = coverage >= self.coverage_threshold
        reason = f"overlap {coverage:.2f}"
        return SentenceVerdict(index, sentence, supported, evidence, reason)


class ScriptedJudge(BaseChecker):
    """A judge whose answers are supplied up front, used to test the interface itself."""

    name = "scripted-judge"

    def __init__(self, answers: dict[str, bool], default: bool = True) -> None:
        self.answers = answers
        self.default = default

    def check_sentence(self, index: int, sentence: str, transcript: Transcript) -> SentenceVerdict:
        supported = self.answers.get(sentence.strip(), self.default)
        return SentenceVerdict(index, sentence, supported, [], "scripted")


class LanguageModelJudge(BaseChecker):
    """Optional judge that calls a chat completion endpoint. Off by default.

    Configuration comes from the environment so that no credential is ever written into the
    repository or printed by the command line tool:

        GROUNDING_JUDGE_URL    the endpoint to post to
        GROUNDING_JUDGE_API_KEY  the bearer token
        GROUNDING_JUDGE_MODEL  the model name to request

    With any of those missing the judge raises `JudgeNotConfigured` before opening a socket.
    """

    name = "llm-judge"

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 30.0):
        self.url = url or os.environ.get(URL_ENV, "")
        self.model = model or os.environ.get(MODEL_ENV, "")
        self.timeout = timeout

    def _api_key(self) -> str:
        return os.environ.get(KEY_ENV, "")

    def ensure_configured(self) -> None:
        required = (
            (URL_ENV, self.url),
            (MODEL_ENV, self.model),
            (KEY_ENV, self._api_key()),
        )
        missing = [name for name, value in required if not value]
        if missing:
            raise JudgeNotConfigured(
                "the optional language model judge is not configured; set "
                + ", ".join(missing)
                + " to enable it"
            )

    def build_prompt(self, sentence: str, transcript: Transcript) -> str:
        lines = [f"{t.number}. {t.speaker}: {t.text}" for t in transcript.turns]
        return (
            "Decide whether the sentence is supported by the conversation.\n"
            'Answer with JSON: {"supported": true or false, "turns": [turn numbers]}.\n\n'
            "Conversation:\n" + "\n".join(lines) + f"\n\nSentence: {sentence}\n"
        )

    def check_sentence(self, index: int, sentence: str, transcript: Transcript) -> SentenceVerdict:
        self.ensure_configured()
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": self.build_prompt(sentence, transcript)}],
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key()}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode())
        answer = json.loads(body["choices"][0]["message"]["content"])
        return SentenceVerdict(
            index,
            sentence,
            bool(answer.get("supported")),
            [int(t) for t in answer.get("turns", [])],
            "language model judge",
        )


# Checkers that are safe to run offline, and the only ones the evaluation uses.
OFFLINE_CHECKERS = {
    DeterministicChecker.name: DeterministicChecker,
    MockLexicalJudge.name: MockLexicalJudge,
}


def build_checker(name: str):
    """Build a checker by name. The optional judge must be asked for explicitly."""
    if name in OFFLINE_CHECKERS:
        return OFFLINE_CHECKERS[name]()
    if name == LanguageModelJudge.name:
        return LanguageModelJudge()
    raise KeyError(f"unknown checker: {name}")
