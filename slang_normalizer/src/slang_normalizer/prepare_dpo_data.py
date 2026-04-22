from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MLX DPO preference data from debate results without "
            "overwriting earlier runs unless you choose the same output paths."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEBATE_RESULTS_PATH,
        help="Path to the debate JSONL results used to construct preferences.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DPO_TRAIN_PATH,
        help="Where to write the DPO training JSONL file.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=DPO_VALID_PATH,
        help="Where to write the DPO validation JSONL file.",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=VALID_SIZE,
        help="How many examples to reserve for validation.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("worst", "hard_negative"),
        default="hard_negative",
        help=(
            "How to pick the rejected response. 'hard_negative' prefers a "
            "plausible but worse candidate, which is usually safer for DPO."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Skip preference pairs whose chosen/rejected responses are shorter.",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=1.8,
        help=(
            "Skip pairs where chosen and rejected lengths are too different. "
            "This reduces the incentive to learn very short generic outputs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for the train/valid split.",
    )
    return parser.parse_args()


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
        "If the sentence is already formal English, return it unchanged.\n"
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


def word_count(text: str) -> int:
    return len(normalize(text).split())


def length_ratio(left: str, right: str) -> float:
    left_words = max(1, word_count(left))
    right_words = max(1, word_count(right))
    longer = max(left_words, right_words)
    shorter = min(left_words, right_words)
    return longer / shorter


def is_noisy_candidate(text: str) -> bool:
    lowered = text.lower()
    if "(no response)" in lowered:
        return True
    if "<|" in text:
        return True
    if repetition_score(text) >= 1.0:
        return True
    if re.search(r"(!\s*){2,}", text):
        return True
    return False


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


def choose_rejected(record: dict[str, str], selection_mode: str) -> str:
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
    clean_ranked: list[tuple[float, str]] = []

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

        if is_noisy_candidate(candidate):
            continue

        hard_negative_score = 0.0
        hard_negative_score += similarity(candidate, ground_truth) * 2.0
        hard_negative_score += similarity(candidate, chosen)
        hard_negative_score += critique_penalty(critique, label) * 0.5
        hard_negative_score -= similarity(candidate, original) * 0.35
        hard_negative_score -= repetition_score(candidate) * 0.5
        clean_ranked.append((hard_negative_score, candidate))

    if selection_mode == "hard_negative" and clean_ranked:
        clean_ranked.sort(key=lambda item: item[0], reverse=True)
        return clean_ranked[0][1]

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
    args = parse_args()
    debate_records = load_jsonl(args.input_path)
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
        rejected = choose_rejected(debate_record, args.selection_mode).strip()

        if not chosen or not rejected:
            continue
        if normalize(chosen) == normalize(rejected):
            continue
        if word_count(chosen) < args.min_words or word_count(rejected) < args.min_words:
            continue
        if length_ratio(chosen, rejected) > args.max_length_ratio:
            continue
        if is_noisy_candidate(chosen):
            continue

        dpo_records.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    if len(dpo_records) <= args.valid_size:
        raise ValueError(
            f"Need more than {args.valid_size} DPO examples, but only built "
            f"{len(dpo_records)}."
        )

    shuffled = list(dpo_records)
    random.Random(args.seed).shuffle(shuffled)
    valid_records = shuffled[: args.valid_size]
    train_records = shuffled[args.valid_size :]

    write_jsonl(train_records, args.train_output)
    write_jsonl(valid_records, args.valid_output)

    print(f"Loaded {len(debate_records)} debate results from {args.input_path}")
    print(f"Built {len(dpo_records)} usable DPO preference pairs")
    print(f"Rejected selection mode: {args.selection_mode}")
    print(f"Wrote {len(train_records)} training pairs to {args.train_output}")
    print(f"Wrote {len(valid_records)} validation pairs to {args.valid_output}")


if __name__ == "__main__":
    main()
