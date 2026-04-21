from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import generate, load
from openai import OpenAI

from slang_normalizer.prepare_mlx_data import (
    BASELINE_HOLDOUT_SIZE,
    PREPROCESSED_DATA_PATH,
    attach_metadata,
    load_preprocessed_records,
    rebuild_records_with_metadata,
)
from slang_normalizer.train_mlx import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    ensure_model_access,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "results_advanced.jsonl"
MAX_RETRIES = 1
JUDGE_MODEL = "deepseek-chat"

JUDGE_SYSTEM_PROMPT = (
    "You are an expert linguist. Compare the 'Formal Translation' with the "
    "'Original Meaning'.\n"
    "Does the translation preserve the core factual meaning?\n"
    "Note: Since it is translating slang to formal English, a slight loss of "
    "informal emotional tone is acceptable.\n"
    "Reply ONLY with a valid JSON in this format:\n"
    '{"status": "PASS" or "FAIL", "feedback": "Brief reason for failure if any"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the advanced back-translation and self-correction pipeline on "
            "the 500 baseline holdout examples."
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
        help="Where to write the advanced pipeline outputs in JSONL format.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each local response.",
    )
    return parser.parse_args()


def load_test_records(limit: int) -> list[dict[str, str | int]]:
    preprocessed_records = load_preprocessed_records(PREPROCESSED_DATA_PATH)
    rebuilt_records = rebuild_records_with_metadata()
    enriched_records = attach_metadata(preprocessed_records, rebuilt_records)
    return enriched_records[:limit]


def build_translation_prompt(
    record: dict[str, str | int], feedback: str | None = None
) -> str:
    prompt = (
        "### Instruction: Given this is a slang from the "
        f"{record['region']} in {record['year']}, rewrite the following sentence "
        "in formal English.\n\n"
        f"### Input: {record['input']}\n\n"
        "### Response:"
    )

    if feedback:
        prompt += (
            "\n\nYour previous attempt failed. "
            f"Feedback: {feedback}. "
            "Please rewrite it to perfectly match the original meaning."
        )

    return prompt


def clean_local_generation(text: str) -> str:
    cleaned = text.strip()

    if "### Response:" in cleaned:
        cleaned = cleaned.split("### Response:", maxsplit=1)[-1].strip()

    cleaned = re.split(r"\n###\s", cleaned, maxsplit=1)[0].strip()
    return cleaned


def generate_local_translation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
) -> str:
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return clean_local_generation(output)


def build_verification_payload(original_meaning: str, formal_translation: str) -> str:
    return (
        f"Original Meaning: {original_meaning}\n"
        f"Formal Translation: {formal_translation}"
    )


def parse_verifier_response(content: str) -> dict[str, str]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return {
                "status": "FAIL",
                "feedback": (
                    "Verifier returned non-JSON output and it could not be parsed."
                ),
            }

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "status": "FAIL",
                "feedback": (
                    "Verifier returned malformed JSON and it could not be parsed."
                ),
            }

    status = str(parsed.get("status", "FAIL")).upper().strip()
    feedback = str(parsed.get("feedback", "")).strip()

    if status not in {"PASS", "FAIL"}:
        return {
            "status": "FAIL",
            "feedback": "Verifier returned an invalid status value.",
        }

    if status == "PASS":
        feedback = ""

    return {"status": status, "feedback": feedback}


def verify_translation(
    client: OpenAI, original_meaning: str, formal_translation: str
) -> dict[str, str]:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_verification_payload(
                    original_meaning=original_meaning,
                    formal_translation=formal_translation,
                ),
            },
        ],
    )

    content = response.choices[0].message.content or ""
    return parse_verifier_response(content)


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in the repository root .env file before running "
            "the advanced back-translation pipeline."
        )

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
    test_records = load_test_records(BASELINE_HOLDOUT_SIZE)

    results: list[dict[str, str]] = []

    for index, record in enumerate(test_records, start=1):
        original_slang = str(record["input"])
        ground_truth = str(record["output"])

        initial_prompt = build_translation_prompt(record)
        initial_translation = generate_local_translation(
            model=model,
            tokenizer=tokenizer,
            prompt=initial_prompt,
            max_tokens=args.max_tokens,
        )

        verification = verify_translation(
            client=client,
            original_meaning=ground_truth,
            formal_translation=initial_translation,
        )
        verification_status = verification["status"]
        feedback = verification["feedback"]
        final_translation = initial_translation

        if verification_status == "PASS":
            print(f"Entry {index}: PASS")
        else:
            print(f"Entry {index}: FAIL -> Correcting...")
            correction_prompt = build_translation_prompt(record, feedback=feedback)

            for _ in range(MAX_RETRIES):
                final_translation = generate_local_translation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=correction_prompt,
                    max_tokens=args.max_tokens,
                )

        results.append(
            {
                "original_slang": original_slang,
                "ground_truth": ground_truth,
                "initial_translation": initial_translation,
                "verification_status": verification_status,
                "feedback": feedback,
                "final_translation": final_translation,
            }
        )

    write_jsonl(results, args.output_path)
    print(f"Wrote {len(results)} advanced results to {args.output_path}")


if __name__ == "__main__":
    main()
