from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors

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
        description="Run MLX inference on the 200 baseline holdout examples."
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
        "### Instruction:\n"
        "Rewrite the following slang sentence in clear formal English.\n"
        "Preserve the original meaning. Do not add new facts.\n"
        "If the sentence is already formal English, return it unchanged.\n"
        "Output only the rewritten sentence.\n\n"
        "### Context:\n"
        f"Region: {record['region']}\n"
        f"Year: {record['year']}\n\n"
        "### Input:\n"
        f"{record['input']}\n\n"
        "### Response:\n"
    )


def clean_generation(text: str) -> str:
    cleaned = text.strip()

    if "### Response:" in cleaned:
        cleaned = cleaned.split("### Response:", maxsplit=1)[-1].strip()

    cleaned = re.split(r"\n###\s", cleaned, maxsplit=1)[0].strip()
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"[!?.;,:\-]*\s*<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"(!\s*){2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
    logits_processors = make_logits_processors(
        repetition_penalty=1.1,
        repetition_context_size=64,
        presence_penalty=0.1,
        presence_context_size=64,
    )

    for index, record in enumerate(test_records, start=1):
        prompt = build_prompt(record)
        raw_prediction = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=min(args.max_tokens, 64),
            logits_processors=logits_processors,
            verbose=False,
        )
        prediction = clean_generation(raw_prediction)

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
