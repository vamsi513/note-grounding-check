import pytest

from grounding.base import SentenceChecker
from grounding.checker import DeterministicChecker
from grounding.judges import (
    KEY_ENV,
    MODEL_ENV,
    URL_ENV,
    JudgeNotConfigured,
    LanguageModelJudge,
    MockLexicalJudge,
    ScriptedJudge,
    build_checker,
)


def test_both_offline_checkers_satisfy_the_interface():
    for checker in (DeterministicChecker(), MockLexicalJudge(), ScriptedJudge({})):
        assert isinstance(checker, SentenceChecker)


def test_a_judge_can_be_swapped_in_through_the_interface(refill_transcript):
    judge = ScriptedJudge({"Patient denies dizziness.": False})
    result = judge.check("Patient denies dizziness. A refill was sent.", refill_transcript, "n")
    assert [v.supported for v in result.sentences] == [False, True]
    assert result.checker == "scripted-judge"


def test_the_mock_judge_misses_a_dose_change_that_the_checker_catches(refill_transcript):
    sentence = "Dose is lisinopril 20 mg once daily."
    assert MockLexicalJudge().check_sentence(0, sentence, refill_transcript).supported
    assert not DeterministicChecker().check_sentence(0, sentence, refill_transcript).supported


def test_the_registry_only_offers_offline_checkers():
    assert sorted(["deterministic", "mock-judge"]) == sorted(
        name for name in ("deterministic", "mock-judge") if build_checker(name)
    )
    with pytest.raises(KeyError):
        build_checker("no-such-checker")


def test_the_model_judge_is_not_in_the_offline_registry():
    from grounding.judges import OFFLINE_CHECKERS

    assert LanguageModelJudge.name not in OFFLINE_CHECKERS


def test_the_model_judge_refuses_to_run_unconfigured(monkeypatch, refill_transcript):
    for name in (URL_ENV, MODEL_ENV, KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    judge = LanguageModelJudge()
    with pytest.raises(JudgeNotConfigured) as excinfo:
        judge.check_sentence(0, "A refill was sent.", refill_transcript)
    assert URL_ENV in str(excinfo.value)


def test_the_model_judge_error_names_every_missing_setting(monkeypatch):
    monkeypatch.setenv(URL_ENV, "https://example.invalid/v1/chat")
    monkeypatch.delenv(MODEL_ENV, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    judge = LanguageModelJudge()
    with pytest.raises(JudgeNotConfigured) as excinfo:
        judge.ensure_configured()
    message = str(excinfo.value)
    assert MODEL_ENV in message
    assert KEY_ENV in message
    assert URL_ENV not in message


def test_the_model_judge_prompt_carries_the_numbered_turns(refill_transcript):
    prompt = LanguageModelJudge(url="x", model="y").build_prompt(
        "A refill was sent.", refill_transcript
    )
    assert "1. doctor: What brings you in today?" in prompt
    assert "A refill was sent." in prompt


def test_the_model_judge_never_puts_a_credential_in_the_prompt(monkeypatch, refill_transcript):
    monkeypatch.setenv(KEY_ENV, "secret-value-should-not-appear")
    prompt = LanguageModelJudge(url="x", model="y").build_prompt("A note.", refill_transcript)
    assert "secret-value-should-not-appear" not in prompt
