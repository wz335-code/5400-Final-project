from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import generate, load

from slang_normalizer.prepare_mlx_data import (
    BASELINE_HOLDOUT_SIZE,
    PREPROCESSED_DATA_PATH,
    attach_metadata,
    load_preprocessed_records,
    rebuild_records_with_metadata,
)
from slang_normalizer.train_mlx import DEFAULT_ADAPTER_PATH, DEFAULT_MODEL

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "results_finetuned.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MLX inference on the 500 baseline holdout examples."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face repo id or local path for the MLX base model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Directory containing the fine-tuned MLX adapter weights.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to write the fine-tuned model outputs in JSONL format.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per example.",
    )
    return parser.parse_args()


def load_test_records(limit: int) -> list[dict[str, str | int]]:
    preprocessed_records = load_preprocessed_records(PREPROCESSED_DATA_PATH)
    rebuilt_records = rebuild_records_with_metadata()
    enriched_records = attach_metadata(preprocessed_records, rebuilt_records)
    return enriched_records[:limit]


def build_prompt(record: dict[str, str | int]) -> str:
    return (
        "### Instruction: Given this is a slang from the "
        f"{record['region']} in {record['year']}, rewrite the following sentence "
        "in formal English.\n\n"
        f"### Input: {record['input']}\n\n"
        "### Response:"
    )


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    test_records = load_test_records(BASELINE_HOLDOUT_SIZE)

    results: list[dict[str, str]] = []

    for index, record in enumerate(test_records, start=1):
        prompt = build_prompt(record)
        prediction = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        ).strip()

        results.append(
            {
                "original_sentence": str(record["input"]),
                "ground_truth": str(record["output"]),
                "finetuned_output": prediction,
            }
        )

        print(f"Completed {index}/{len(test_records)} examples")

    write_jsonl(results, args.output_path)
    print(f"Wrote {len(results)} fine-tuned results to {args.output_path}")


if __name__ == "__main__":
    main()
