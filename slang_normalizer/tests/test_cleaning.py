from slang_normalizer.preprocess import clean_text


def test_clean_text_normalizes_case_punctuation_and_repetitions() -> None:
    text = "Sooo!!! That's, WILD..."

    assert clean_text(text) == "so that s wild"
