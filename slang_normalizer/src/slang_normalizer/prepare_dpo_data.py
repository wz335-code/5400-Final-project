from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

from slang_normalizer.prepare_mlx_data import (
    PREPROCESSED_DATA_PATH,
    attach_metadata,
    load_preprocessed_records,
    rebuild_records_with_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEBATE_RESULTS_PATH = REPO_ROOT / "data" / "results_debate.jsonl"
DPO_TRAIN_PATH = REPO_ROOT / "data" / "dpo_train.jsonl"
DPO_VALID_PATH = REPO_ROOT / "data" / "dpo_valid.jsonl"
VALID_SIZE = 20
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


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
        "Output only the rewritten sentence.\n\n"
        "### Context:\n"
        f"Region: {record['region']}\n"
        f"Year: {record['year']}\n\n"
        "### Input:\n"
        f"{record['input']}\n\n"
        "### Response:\n"
    )


def normalize(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def repetition_score(text: str) -> float:
    tokens = normalize(text).split()
    if not tokens:
        return 3.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return max(0.0, 1.0 - unique_ratio) * 3.0


def critique_penalty(critique: str, label: str) -> float:
    critique_lower = critique.lower()
    marker = f"candidate {label}"
    negative_markers = (
        "fail",
        "incorrect",
        "awkward",
        "repetitive",
        "identical",
        "retains",
        "vulgar",
        "echo",
        "no translation",
        "unprofessional",
    )
    if marker in critique_lower and any(
        word in critique_lower for word in negative_markers
    ):
        return 1.0
    return 0.0


def choose_rejected(record: dict[str, str]) -> str:
    chosen = record["final_best_translation"]
    original = record["original_slang"]
    ground_truth = record["ground_truth"]
    critique = record["critique"]

    candidates = {
        "a": record["candidate_a"],
        "b": record["candidate_b"],
        "c": record["candidate_c"],
    }

    ranked: list[tuple[float, str]] = []

    for label, candidate in candidates.items():
        if not candidate or normalize(candidate) == normalize(chosen):
            continue

        score = 0.0
        if normalize(candidate) == normalize(original):
            score += 3.0
        if "(no response)" in candidate.lower():
            score += 3.0
        if "<|" in candidate:
            score += 2.0

        score += repetition_score(candidate)
        score += (1.0 - similarity(candidate, ground_truth)) * 2.0
        score += similarity(candidate, original)
        score += critique_penalty(critique, label)
        ranked.append((score, candidate))

    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    return original


def write_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    debate_records = load_jsonl(DEBATE_RESULTS_PATH)
    metadata_records = load_test_records(len(debate_records))

    if len(debate_records) != len(metadata_records):
        raise ValueError(
            "results_debate.jsonl does not align with the reconstructed metadata "
            "records."
        )

    dpo_records: list[dict[str, str]] = []

    for metadata_record, debate_record in zip(
        metadata_records, debate_records, strict=True
    ):
        if normalize(str(metadata_record["input"])) != normalize(
            debate_record["original_slang"]
        ) or normalize(str(metadata_record["output"])) != normalize(
            debate_record["ground_truth"]
        ):
            raise ValueError(
                "Debate results and metadata records are out of sync. "
                "Refusing to build DPO data from mismatched rows."
            )

        prompt = build_prompt(metadata_record)
        chosen = debate_record["final_best_translation"].strip()
        rejected = choose_rejected(debate_record).strip()

        if not chosen or not rejected:
            continue
        if normalize(chosen) == normalize(rejected):
            continue

        dpo_records.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    if len(dpo_records) <= VALID_SIZE:
        raise ValueError(
            f"Need more than {VALID_SIZE} DPO examples, but only built "
            f"{len(dpo_records)}."
        )

    shuffled = list(dpo_records)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    valid_records = shuffled[:VALID_SIZE]
    train_records = shuffled[VALID_SIZE:]

    write_jsonl(train_records, DPO_TRAIN_PATH)
    write_jsonl(valid_records, DPO_VALID_PATH)

    print(f"Loaded {len(debate_records)} debate results from {DEBATE_RESULTS_PATH}")
    print(f"Built {len(dpo_records)} usable DPO preference pairs")
    print(f"Wrote {len(train_records)} training pairs to {DPO_TRAIN_PATH}")
    print(f"Wrote {len(valid_records)} validation pairs to {DPO_VALID_PATH}")


if __name__ == "__main__":
    main()
