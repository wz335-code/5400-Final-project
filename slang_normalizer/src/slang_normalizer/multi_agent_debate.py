"""Run the multi-agent debate pipeline for slang normalization."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from openai import AsyncOpenAI

from slang_normalizer.prepare_mlx_data import (
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
OUTPUT_PATH = REPO_ROOT / "data" / "results_debate.jsonl"
TEST_LIMIT = 200
MAX_CONCURRENCY = 10
JUDGE_MODEL = "deepseek-chat"
# This pipeline compares three local candidates and asks DeepSeek to choose.
DEBATE_SYSTEM_PROMPT = """You are the head judge in a linguistic debate.
Your goal is to translate slang into formal English
without losing the core factual meaning.
You will be provided with the Original Slang,
its Literal Meaning, and 3 Candidate Translations.
Step 1: Critique the pros and cons of Candidate A, B, and C.
Step 2: Based on your critique, synthesize the absolute best formal translation.
You can pick the best candidate, or combine their strengths into a new sentence.
Reply ONLY with a valid JSON in this exact format:
{"critique": "Brief analysis of the candidates",
"final_best_translation": "The synthesized perfect translation"}"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for debate inference."""

    parser = argparse.ArgumentParser(
        description=("Run multi-agent debate inference on the first 200 test examples.")
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
        help="Where to write the debate results in JSONL format.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate for each local candidate.",
    )
    return parser.parse_args()


def load_test_records(limit: int) -> list[dict[str, str | int]]:
    """Load the shared holdout set with metadata."""

    preprocessed_records = load_preprocessed_records(PREPROCESSED_DATA_PATH)
    rebuilt_records = rebuild_records_with_metadata()
    enriched_records = attach_metadata(preprocessed_records, rebuilt_records)
    return enriched_records[:limit]


def build_brainstorm_prompt(record: dict[str, str | int]) -> str:
    """Build the prompt used for each local candidate."""

    return (
        "### Instruction:\n"
        "Rewrite the following slang sentence in clear formal English.\n"
        "Preserve the original meaning. Do not add new facts.\n"
        "Produce one fluent formal rewrite only.\n\n"
        "### Context:\n"
        f"Region: {record['region']}\n"
        f"Year: {record['year']}\n\n"
        "### Input:\n"
        f"{record['input']}\n\n"
        "### Response:\n"
    )


def clean_local_generation(text: str) -> str:
    """Clean the raw local candidate output."""

    cleaned = text.strip()

    if "### Response:" in cleaned:
        cleaned = cleaned.split("### Response:", maxsplit=1)[-1].strip()

    cleaned = re.split(r"\n###\s", cleaned, maxsplit=1)[0].strip()
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"[!?.;,:\-]*\s*<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"(!\s*){2,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_candidate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    sampler,
    logits_processors,
) -> str:
    """Generate one local candidate translation."""

    # Use sampling so the three candidates are not identical.
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=min(max_tokens, 64),
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    return clean_local_generation(output)


def build_judge_payload(record: dict[str, str | int]) -> str:
    """Build the judge payload with three candidate translations."""

    return (
        f"Original Slang: {record['original_slang']}\n"
        f"Literal Meaning: {record['ground_truth']}\n"
        f"Candidate A: {record['candidate_a']}\n"
        f"Candidate B: {record['candidate_b']}\n"
        f"Candidate C: {record['candidate_c']}"
    )


def parse_judge_response(
    content: str, fallback_record: dict[str, str]
) -> dict[str, str]:
    """Parse the judge response into critique and final translation."""

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, dict):
        critique = str(parsed.get("critique", "")).strip()
        final_best_translation = str(parsed.get("final_best_translation", "")).strip()
        if final_best_translation:
            return {
                "critique": (
                    critique or "Parsed JSON successfully, but critique was empty."
                ),
                "final_best_translation": final_best_translation,
            }

    fallback_translation = (
        fallback_record["candidate_a"]
        or fallback_record["candidate_b"]
        or fallback_record["candidate_c"]
        or fallback_record["ground_truth"]
    )
    fallback_critique = (
        "DeepSeek returned malformed JSON; fell back to Candidate A. "
        f"Raw response: {content.strip()}"
    )
    return {
        "critique": fallback_critique,
        "final_best_translation": fallback_translation,
    }


async def judge_and_synthesize(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    record: dict[str, str],
) -> dict[str, str]:
    """Ask DeepSeek to critique candidates and synthesize the best answer."""

    # DeepSeek critiques the candidates and writes one final answer.
    async with semaphore:
        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": DEBATE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_payload(record)},
            ],
        )

    content = response.choices[0].message.content or ""
    judged = parse_judge_response(content, record)

    return {
        "original_slang": record["original_slang"],
        "ground_truth": record["ground_truth"],
        "candidate_a": record["candidate_a"],
        "candidate_b": record["candidate_b"],
        "candidate_c": record["candidate_c"],
        "critique": judged["critique"],
        "final_best_translation": judged["final_best_translation"],
    }


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    """Write debate results to a JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main() -> None:
    """Run multi-agent debate inference on the shared holdout set."""

    args = parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in the repository root .env file before running "
            "the multi-agent debate pipeline."
        )

    ensure_model_access(args.model)

    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    test_records = load_test_records(TEST_LIMIT)
    logits_processors = make_logits_processors(
        repetition_penalty=1.1,
        repetition_context_size=64,
        presence_penalty=0.1,
        presence_context_size=64,
    )
    sampler = make_sampler(temp=0.7, top_p=0.9)

    candidate_records: list[dict[str, str]] = []

    for index, record in enumerate(test_records, start=1):
        # Local generation stays sequential, then API judging runs in parallel.
        prompt = build_brainstorm_prompt(record)

        candidate_a = generate_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )
        candidate_b = generate_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )
        candidate_c = generate_candidate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        candidate_records.append(
            {
                "original_slang": str(record["input"]),
                "ground_truth": str(record["output"]),
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "candidate_c": candidate_c,
            }
        )
        print(f"Prepared local candidates for entry {index}/{len(test_records)}")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    try:
        results = await asyncio.gather(
            *[
                judge_and_synthesize(client, semaphore, record)
                for record in candidate_records
            ]
        )
    finally:
        await client.close()

    write_jsonl(results, args.output_path)
    print(f"Wrote {len(results)} debate results to {args.output_path}")


def run() -> None:
    """Run the async debate entry point from a console script."""

    asyncio.run(main())


if __name__ == "__main__":
    run()
