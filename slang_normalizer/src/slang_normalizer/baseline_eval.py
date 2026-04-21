from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

PROMPT = (
    "Identify the slang in this sentence and rewrite it in formal English. "
    "Output only the formal sentence, nothing else."
)
REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = REPO_ROOT / "data" / "slang_open_sub_llama3.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "results_baseline.jsonl"
MAX_EXAMPLES = 500
MAX_CONCURRENCY = 5


def load_examples(input_path: Path, limit: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if len(examples) >= limit:
                break

            record = json.loads(line)
            examples.append(
                {
                    "original_sentence": record["input"],
                    "ground_truth": record["output"],
                }
            )

    return examples


async def rewrite_sentence(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    example: dict[str, str],
) -> dict[str, str]:
    async with semaphore:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{PROMPT}\n\nSentence: {example['original_sentence']}"
                    ),
                }
            ],
        )

    deepseek_output = response.choices[0].message.content or ""

    return {
        "original_sentence": example["original_sentence"],
        "ground_truth": example["ground_truth"],
        "deepseek_output": deepseek_output.strip(),
    }


def write_results(results: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")


async def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "Set DEEPSEEK_API_KEY in the repository root .env file before running "
            "the baseline evaluation."
        )

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    examples = load_examples(INPUT_PATH, MAX_EXAMPLES)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    try:
        results = await asyncio.gather(
            *[rewrite_sentence(client, semaphore, example) for example in examples]
        )
    finally:
        await client.close()

    write_results(results, OUTPUT_PATH)
    print(f"Wrote {len(results)} baseline results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
