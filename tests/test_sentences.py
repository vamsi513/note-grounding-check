from grounding.sentences import split_sentences


def test_splits_on_full_stops():
    text = "Patient denies cough. Lungs sound clear. A refill was sent."
    assert split_sentences(text) == [
        "Patient denies cough.",
        "Lungs sound clear.",
        "A refill was sent.",
    ]


def test_decimal_dose_does_not_split():
    assert split_sentences("Patient takes warfarin 2.5 mg daily.") == [
        "Patient takes warfarin 2.5 mg daily."
    ]


def test_two_decimals_in_one_sentence_stay_together():
    text = "Creatinine is 1.3, up from 1.1 last year. A recheck is planned."
    assert split_sentences(text) == [
        "Creatinine is 1.3, up from 1.1 last year.",
        "A recheck is planned.",
    ]


def test_abbreviation_does_not_split():
    assert split_sentences("Seen by Dr. Lane for a refill.") == ["Seen by Dr. Lane for a refill."]


def test_question_and_exclamation_end_sentences():
    assert split_sentences("Is the dose right? It is 10 mg!") == [
        "Is the dose right?",
        "It is 10 mg!",
    ]


def test_newlines_and_runs_of_space_are_collapsed():
    assert split_sentences("First line.\n\n  Second   line.") == ["First line.", "Second line."]


def test_empty_input_gives_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []


def test_trailing_text_without_a_full_stop_is_kept():
    assert split_sentences("A refill was sent. No follow up needed") == [
        "A refill was sent.",
        "No follow up needed",
    ]
