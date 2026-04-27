"""Run a 5-sentence demo of the advanced self-correction pipeline."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import load
from mlx_lm.sample_utils import make_logits_processors
from openai import OpenAI

from slang_normalizer.advanced_with_backtranslation import (
    MAX_RETRIES,
    build_correction_prompt,
    build_initial_prompt,
    generate_local_translation,
    load_test_records,
    verification_rank,
    verify_translation,
    write_jsonl,
)
from slang_normalizer.logging_utils import configure_logging
from slang_normalizer.train_mlx import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    ensure_model_access,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "results_advanced_demo_5.jsonl"
DEMO_SIZE = 5
# Small demo script: same advanced logic, but only a few examples.
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the 5-sentence demo."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a small 5-sentence demo of the advanced back-translation and "
            "self-correction pipeline."
        )
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
        help="Where to write the 5-sentence demo outputs in JSONL format.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each local response.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEMO_SIZE,
        help="How many test samples to translate in this demo.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a small advanced demo and print results in the terminal."""

    configure_logging()
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in the repository root .env file before "
            "running the advanced demo pipeline."
        )

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")

    ensure_model_access(args.model)

    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    test_records = load_test_records(args.num_samples)
    logger.info("Loaded %d demo samples", len(test_records))
    logits_processors = make_logits_processors(
        repetition_penalty=1.1,
        repetition_context_size=64,
        presence_penalty=0.1,
        presence_context_size=64,
    )

    results: list[dict[str, str]] = []

    for index, record in enumerate(test_records, start=1):
        # Reuse the advanced pipeline on a much smaller sample set.
        original_slang = str(record["input"])
        ground_truth = str(record["output"])

        initial_translation = generate_local_translation(
            model=model,
            tokenizer=tokenizer,
            prompt=build_initial_prompt(record),
            max_tokens=args.max_tokens,
            logits_processors=logits_processors,
        )

        verification = verify_translation(
            client=client,
            original_meaning=ground_truth,
            formal_translation=initial_translation,
        )
        best_translation = initial_translation
        best_status = verification["status"]
        best_feedback = verification["feedback"]

        if best_status == "PASS":
            print(f"Demo entry {index}: PASS")
        else:
            print(f"Demo entry {index}: {best_status} -> Correcting...")
            correction_prompt = build_correction_prompt(
                record=record,
                previous_attempt=initial_translation,
                feedback=best_feedback,
                severity=best_status,
            )

            for _ in range(MAX_RETRIES):
                candidate_translation = generate_local_translation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=correction_prompt,
                    max_tokens=args.max_tokens,
                    logits_processors=logits_processors,
                )
                candidate_verification = verify_translation(
                    client=client,
                    original_meaning=ground_truth,
                    formal_translation=candidate_translation,
                )

                if verification_rank(candidate_verification["status"]) >= (
                    verification_rank(best_status)
                ):
                    best_translation = candidate_translation
                    best_status = candidate_verification["status"]
                    best_feedback = candidate_verification["feedback"]
                else:
                    print(
                        f"Demo entry {index}: correction was worse "
                        f"({candidate_verification['status']} < {best_status}), "
                        "keeping previous result"
                    )

                if best_status == "PASS":
                    print(f"Demo entry {index}: PASS after correction")
                    break

            if best_status != "PASS":
                print(f"Demo entry {index}: {best_status} after correction")

        results.append(
            {
                "original_slang": original_slang,
                "ground_truth": ground_truth,
                "initial_translation": initial_translation,
                "verification_status": best_status,
                "feedback": best_feedback,
                "final_translation": best_translation,
            }
        )

        print(f"\n=== Demo Result {index} ===")
        print(f"Original Slang: {original_slang}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Initial Translation: {initial_translation}")
        print(f"Judge Status: {best_status}")
        print(f"Feedback: {best_feedback or '(none)'}")
        print(f"Final Translation: {best_translation}")

    write_jsonl(results, args.output_path)
    logger.info("Saved demo outputs to %s", args.output_path)
    print(f"Wrote {len(results)} demo results to {args.output_path}")


if __name__ == "__main__":
    main()
