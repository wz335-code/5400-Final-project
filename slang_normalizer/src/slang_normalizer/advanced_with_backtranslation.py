from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors
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
    "You are an expert linguist evaluating slang normalization.\n"
    "Compare the Formal Translation against the Original Meaning.\n"
    "Use exactly one of these labels:\n"
    "PASS: The translation preserves the core meaning and does not add false "
    "facts. Differences in wording are acceptable. Slight loss of slang tone "
    "is acceptable.\n"
    "SOFT_FAIL: The translation is mostly correct, but wording, nuance, or "
    "tone could be improved to better match the original meaning.\n"
    "HARD_FAIL: The translation changes meaning, omits important "
    "information, adds unsupported details, or seriously distorts tone.\n"
    "Reply ONLY with valid JSON in this format:\n"
    '{"status": "PASS", "feedback": ""} or '
    '{"status": "SOFT_FAIL", "feedback": "..."} or '
    '{"status": "HARD_FAIL", "feedback": "..."}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the advanced back-translation and self-correction pipeline on "
            "the 200 baseline holdout examples."
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


def build_initial_prompt(record: dict[str, str | int]) -> str:
    return (
        "### Instruction:\n"
        "Rewrite the following slang sentence in clear formal English.\n"
        "Preserve the original meaning. Do not add new facts.\n\n"
        "If the sentence is already formal English, return it unchanged.\n"
        "Output only the rewritten sentence.\n\n"
        "### Context:\n"
        f"Region: {record['region']}\n"
        f"Year: {record['year']}\n\n"
        "### Input:\n"
        f"{record['input']}\n\n"
        "### Response:\n"
    )


def build_correction_prompt(
    record: dict[str, str | int],
    previous_attempt: str,
    feedback: str,
    severity: str,
) -> str:
    return (
        "### Instruction:\n"
        "Rewrite the following slang sentence in clear formal English.\n"
        "Preserve the original meaning. Do not add new facts.\n"
        "Output only the rewritten sentence.\n"
        f"Your previous attempt was rated {severity} and did not fully satisfy "
        "the target meaning.\n\n"
        "### Context:\n"
        f"Region: {record['region']}\n"
        f"Year: {record['year']}\n\n"
        "### Input:\n"
        f"{record['input']}\n\n"
        "### Feedback:\n"
        f"{feedback}\n\n"
        "### Previous Attempt:\n"
        f"{previous_attempt}\n\n"
        "### Response:\n"
    )


def clean_local_generation(text: str) -> str:
    cleaned = text.strip()

    if "### Response:" in cleaned:
        cleaned = cleaned.split("### Response:", maxsplit=1)[-1].strip()

    cleaned = re.split(r"\n###\s", cleaned, maxsplit=1)[0].strip()
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"[!?.;,:\-]*\s*<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"(!\s*){2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_local_translation(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    logits_processors,
) -> str:
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=min(max_tokens, 64),
        logits_processors=logits_processors,
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
                "status": "HARD_FAIL",
                "feedback": (
                    "Verifier returned non-JSON output and it could not be parsed."
                ),
            }

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "status": "HARD_FAIL",
                "feedback": (
                    "Verifier returned malformed JSON and it could not be parsed."
                ),
            }

    status = str(parsed.get("status", "FAIL")).upper().strip()
    feedback = str(parsed.get("feedback", "")).strip()

    if status not in {"PASS", "SOFT_FAIL", "HARD_FAIL"}:
        return {
            "status": "HARD_FAIL",
            "feedback": "Verifier returned an invalid status value.",
        }

    if status == "PASS":
        feedback = ""

    return {"status": status, "feedback": feedback}


def verification_rank(status: str) -> int:
    return {
        "PASS": 3,
        "SOFT_FAIL": 2,
        "HARD_FAIL": 1,
    }.get(status, 0)


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
    logits_processors = make_logits_processors(
        repetition_penalty=1.1,
        repetition_context_size=64,
        presence_penalty=0.1,
        presence_context_size=64,
    )

    results: list[dict[str, str]] = []

    for index, record in enumerate(test_records, start=1):
        original_slang = str(record["input"])
        ground_truth = str(record["output"])

        initial_prompt = build_initial_prompt(record)
        initial_translation = generate_local_translation(
            model=model,
            tokenizer=tokenizer,
            prompt=initial_prompt,
            max_tokens=args.max_tokens,
            logits_processors=logits_processors,
        )

        verification = verify_translation(
            client=client,
            original_meaning=ground_truth,
            formal_translation=initial_translation,
        )
        verification_status = verification["status"]
        feedback = verification["feedback"]
        final_translation = initial_translation
        best_status = verification_status
        best_feedback = feedback
        best_translation = initial_translation

        if verification_status == "PASS":
            print(f"Entry {index}: PASS")
        else:
            print(f"Entry {index}: {verification_status} -> Correcting...")
            correction_prompt = build_correction_prompt(
                record=record,
                previous_attempt=initial_translation,
                feedback=feedback,
                severity=verification_status,
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
                candidate_status = candidate_verification["status"]
                candidate_feedback = candidate_verification["feedback"]

                if verification_rank(candidate_status) >= verification_rank(
                    best_status
                ):
                    best_translation = candidate_translation
                    best_status = candidate_status
                    best_feedback = candidate_feedback
                else:
                    print(
                        f"Entry {index}: correction was worse "
                        f"({candidate_status} < {best_status}), keeping previous result"
                    )

                final_translation = best_translation
                verification_status = best_status
                feedback = best_feedback

                if verification_status == "PASS":
                    print(f"Entry {index}: PASS after correction")
                    break

            if verification_status != "PASS":
                print(f"Entry {index}: {verification_status} after correction")

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
