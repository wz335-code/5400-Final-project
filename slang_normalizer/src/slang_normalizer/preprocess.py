from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

INSTRUCTION = (
    "Normalize the slang in the sentence into clear standard English while "
    "preserving the original meaning."
)

REPO_ROOT = Path(__file__).resolve().parents[3]
POSITIVE_DATA_PATH = REPO_ROOT / "data" / "slang_OpenSub.tsv"
NEGATIVE_DATA_PATH = REPO_ROOT / "data" / "slang_OpenSub_negatives.tsv"
OUTPUT_PATH = REPO_ROOT / "data" / "slang_open_sub_llama3.jsonl"


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation, and collapse elongated spellings."""

    normalized = str(text).lower()
    normalized = re.sub(r"(.)\1{2,}", r"\1", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_slang_mapping(dataframe: pd.DataFrame) -> dict[str, str]:
    """Build a deterministic slang-to-literal mapping from annotated rows."""

    paraphrase_choices: dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in dataframe.dropna(
        subset=["SLANG_TERM", "LITERAL_PARAPHRASE_OF_SLANG"]
    ).iterrows():
        slang_term = clean_text(row["SLANG_TERM"])
        literal_paraphrase = clean_text(row["LITERAL_PARAPHRASE_OF_SLANG"])

        if slang_term and literal_paraphrase:
            paraphrase_choices[slang_term][literal_paraphrase] += 1

    return {
        slang_term: choices.most_common(1)[0][0]
        for slang_term, choices in paraphrase_choices.items()
    }


def normalize_sentence(
    sentence: str, slang_term: str, literal_paraphrase: str
) -> str:
    """Replace the annotated slang span in a cleaned sentence."""

    normalized_sentence = clean_text(sentence)
    pattern = rf"\b{re.escape(slang_term)}\b"
    return re.sub(pattern, literal_paraphrase, normalized_sentence)


def build_positive_examples(
    dataframe: pd.DataFrame, slang_mapping: dict[str, str]
) -> list[dict[str, str]]:
    """Create instruction-tuning examples from positive slang annotations."""

    examples: list[dict[str, str]] = []

    for _, row in dataframe.dropna(subset=["SENTENCE", "SLANG_TERM"]).iterrows():
        cleaned_input = clean_text(row["SENTENCE"])
        slang_term = clean_text(row["SLANG_TERM"])
        literal_paraphrase = slang_mapping.get(slang_term)

        if not slang_term or not literal_paraphrase:
            continue

        cleaned_output = normalize_sentence(
            row["SENTENCE"], slang_term, literal_paraphrase
        )

        if not cleaned_input or not cleaned_output or cleaned_input == cleaned_output:
            continue

        examples.append(
            {
                "instruction": INSTRUCTION,
                "input": cleaned_input,
                "output": cleaned_output,
            }
        )

    return examples


def build_negative_examples(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    """Create identity examples for sentences that should not be normalized."""

    examples: list[dict[str, str]] = []

    for _, row in dataframe.dropna(subset=["SENTENCE"]).iterrows():
        cleaned_sentence = clean_text(row["SENTENCE"])

        if not cleaned_sentence:
            continue

        examples.append(
            {
                "instruction": INSTRUCTION,
                "input": cleaned_sentence,
                "output": cleaned_sentence,
            }
        )

    return examples


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    positive_df = pd.read_csv(POSITIVE_DATA_PATH, sep="\t")
    negative_df = pd.read_csv(NEGATIVE_DATA_PATH, sep="\t")

    slang_mapping = build_slang_mapping(positive_df)
    positive_examples = build_positive_examples(positive_df, slang_mapping)
    negative_examples = build_negative_examples(negative_df)

    all_examples = positive_examples + negative_examples
    write_jsonl(all_examples, OUTPUT_PATH)

    print(f"Loaded {len(positive_df)} positive source rows from {POSITIVE_DATA_PATH}")
    print(f"Loaded {len(negative_df)} negative source rows from {NEGATIVE_DATA_PATH}")
    print(f"Built {len(slang_mapping)} slang mappings")
    print(f"Wrote {len(positive_examples)} positive examples")
    print(f"Wrote {len(negative_examples)} negative examples")
    print(f"Wrote {len(all_examples)} total examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
