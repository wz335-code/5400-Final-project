from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from slang_normalizer.preprocess import (
    NEGATIVE_DATA_PATH,
    POSITIVE_DATA_PATH,
    build_slang_mapping,
    clean_text,
    normalize_sentence,
)
from slang_normalizer.preprocess import (
    OUTPUT_PATH as PREPROCESSED_DATA_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MLX_DATA_DIR = REPO_ROOT / "data" / "mlx"
TRAIN_PATH = MLX_DATA_DIR / "train.jsonl"
VALID_PATH = MLX_DATA_DIR / "valid.jsonl"
BASELINE_HOLDOUT_SIZE = 200
VALID_RATIO = 0.1
RANDOM_SEED = 42


def load_preprocessed_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def build_positive_examples_with_metadata(
    dataframe: pd.DataFrame, slang_mapping: dict[str, str]
) -> list[dict[str, str | int]]:
    examples: list[dict[str, str | int]] = []

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
                "instruction": "Normalize the slang sentence into formal English.",
                "input": cleaned_input,
                "output": cleaned_output,
                "region": str(row["REGION"]).strip(),
                "year": int(row["YEAR"]),
            }
        )

    return examples


def build_negative_examples_with_metadata(
    dataframe: pd.DataFrame,
) -> list[dict[str, str | int]]:
    examples: list[dict[str, str | int]] = []

    for _, row in dataframe.dropna(subset=["SENTENCE"]).iterrows():
        cleaned_sentence = clean_text(row["SENTENCE"])

        if not cleaned_sentence:
            continue

        examples.append(
            {
                "instruction": "Normalize the slang sentence into formal English.",
                "input": cleaned_sentence,
                "output": cleaned_sentence,
                "region": str(row["REGION"]).strip(),
                "year": int(row["YEAR"]),
            }
        )

    return examples


def rebuild_records_with_metadata() -> list[dict[str, str | int]]:
    positive_df = pd.read_csv(POSITIVE_DATA_PATH, sep="\t")
    negative_df = pd.read_csv(NEGATIVE_DATA_PATH, sep="\t")
    slang_mapping = build_slang_mapping(positive_df)

    positive_records = build_positive_examples_with_metadata(positive_df, slang_mapping)
    negative_records = build_negative_examples_with_metadata(negative_df)
    return positive_records + negative_records


def attach_metadata(
    preprocessed_records: list[dict[str, str]],
    rebuilt_records: list[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    if len(preprocessed_records) != len(rebuilt_records):
        raise ValueError(
            "The rebuilt dataset does not match slang_open_sub_llama3.jsonl in "
            "record count."
        )

    enriched_records: list[dict[str, str | int]] = []

    for index, (preprocessed, rebuilt) in enumerate(
        zip(preprocessed_records, rebuilt_records, strict=True),
        start=1,
    ):
        if (
            preprocessed["input"] != rebuilt["input"]
            or preprocessed["output"] != rebuilt["output"]
        ):
            raise ValueError(
                "Dataset alignment failed at line "
                f"{index}: preprocessing output no longer matches the source TSVs."
            )

        enriched_records.append(
            {
                "instruction": preprocessed["instruction"],
                "input": preprocessed["input"],
                "output": preprocessed["output"],
                "region": rebuilt["region"],
                "year": rebuilt["year"],
            }
        )

    return enriched_records


def format_for_mlx(record: dict[str, str | int]) -> dict[str, str]:
    return {
        "text": (
            "### Instruction: Given this is slang from the "
            f"{record['region']} region in {record['year']}, rewrite the "
            "following sentence in formal English.\n\n"
            f"### Input: {record['input']}\n\n"
            f"### Response: {record['output']}"
        )
    }


def split_records(
    records: list[dict[str, str | int]], valid_ratio: float, seed: int
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    shuffled_records = list(records)
    random.Random(seed).shuffle(shuffled_records)

    valid_size = max(1, int(len(shuffled_records) * valid_ratio))
    train_size = len(shuffled_records) - valid_size

    return shuffled_records[:train_size], shuffled_records[train_size:]


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    preprocessed_records = load_preprocessed_records(PREPROCESSED_DATA_PATH)

    if len(preprocessed_records) <= BASELINE_HOLDOUT_SIZE:
        raise ValueError(
            "Not enough preprocessed examples to exclude the baseline holdout split."
        )

    rebuilt_records = rebuild_records_with_metadata()
    enriched_records = attach_metadata(preprocessed_records, rebuilt_records)
    training_records = enriched_records[BASELINE_HOLDOUT_SIZE:]
    train_records, valid_records = split_records(
        training_records,
        valid_ratio=VALID_RATIO,
        seed=RANDOM_SEED,
    )

    formatted_train_records = [format_for_mlx(record) for record in train_records]
    formatted_valid_records = [format_for_mlx(record) for record in valid_records]

    write_jsonl(formatted_train_records, TRAIN_PATH)
    write_jsonl(formatted_valid_records, VALID_PATH)

    print(
        f"Reserved the first {BASELINE_HOLDOUT_SIZE} records for baseline evaluation."
    )
    print(f"Wrote {len(formatted_train_records)} training examples to {TRAIN_PATH}")
    print(f"Wrote {len(formatted_valid_records)} validation examples to {VALID_PATH}")


if __name__ == "__main__":
    main()
