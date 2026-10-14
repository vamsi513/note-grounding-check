from grounding.checker import DeterministicChecker
from grounding.evidence import Quantity, extract_quantities, quantity_satisfied


def test_dose_binds_to_its_unit():
    assert extract_quantities("lisinopril 10 mg once daily") == [
        Quantity("10", "mg"),
        Quantity("1", "daily"),
    ]


def test_number_words_become_quantities():
    assert extract_quantities("continue for eight more weeks") == [Quantity("8", "week")]


def test_decimal_values_are_kept_whole():
    assert extract_quantities("warfarin 2.5 mg") == [Quantity("2.5", "mg")]


def test_number_without_a_unit_has_no_unit():
    assert extract_quantities("readings around 134 over 86") == [
        Quantity("134", None),
        Quantity("86", None),
    ]


def test_a_quantity_matches_itself_only():
    haystack = [Quantity("10", "mg")]
    assert quantity_satisfied(Quantity("10", "mg"), haystack)
    assert not quantity_satisfied(Quantity("20", "mg"), haystack)


def test_a_bare_number_matches_the_same_number_with_a_unit():
    assert quantity_satisfied(Quantity("10", None), [Quantity("10", "mg")])


def test_changed_dose_is_unsupported(refill_transcript):
    checker = DeterministicChecker()
    verdict = checker.check_sentence(0, "Dose is lisinopril 20 mg once daily.", refill_transcript)
    assert not verdict.supported
    assert "20 mg" in verdict.reason


def test_matching_dose_is_supported(refill_transcript):
    checker = DeterministicChecker()
    verdict = checker.check_sentence(0, "Dose is lisinopril 10 mg once daily.", refill_transcript)
    assert verdict.supported


def test_changed_duration_is_unsupported(refill_transcript):
    checker = DeterministicChecker()
    verdict = checker.check_sentence(0, "A 30 day supply was sent.", refill_transcript)
    assert not verdict.supported
    assert "30 day" in verdict.reason
