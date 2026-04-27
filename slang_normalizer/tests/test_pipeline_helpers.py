from slang_normalizer.combine_results import validate_alignment
from slang_normalizer.prepare_mlx_data import format_for_mlx
from slang_normalizer.preprocess import normalize_sentence


def test_normalize_sentence_replaces_the_annotated_slang() -> None:
    """The sentence normalizer should replace the marked slang span."""

    result = normalize_sentence(
        "He will barge in soon.",
        slang_term="barge in",
        literal_paraphrase="intrude",
    )

    assert result == "he will intrude soon"


def test_format_for_mlx_includes_metadata_and_output() -> None:
    """MLX formatting should keep metadata, input text, and target output."""

    record = {
        "instruction": "Normalize the slang sentence into formal English.",
        "input": "you not got a bottle man",
        "output": "you lack courage sir",
        "region": "UK",
        "year": 2006,
    }

    formatted = format_for_mlx(record)

    assert "UK region in 2006" in formatted["text"]
    assert "### Input: you not got a bottle man" in formatted["text"]
    assert "### Response: you lack courage sir" in formatted["text"]


def test_validate_alignment_accepts_matching_rows() -> None:
    """Aligned rows from different result files should pass validation."""

    validate_alignment(
        1,
        {"original_sentence": "a", "ground_truth": "b", "deepseek_output": "c"},
        {"original_sentence": "a", "ground_truth": "b", "finetuned_output": "c"},
        {"original_slang": "a", "ground_truth": "b", "final_translation": "c"},
        {
            "original_slang": "a",
            "ground_truth": "b",
            "final_best_translation": "c",
        },
        {"original_sentence": "a", "ground_truth": "b", "finetuned_output": "c"},
        {"original_slang": "a", "ground_truth": "b", "final_translation": "c"},
    )
