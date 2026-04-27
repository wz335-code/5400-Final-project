"""Merge all experiment outputs into one evaluation-ready dataset."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from slang_normalizer.logging_utils import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "results_combined_for_evaluation.jsonl"
CSV_OUTPUT_PATH = DATA_DIR / "results_combined_for_evaluation.csv"
# Merge all model outputs into one evaluation table.
logger = logging.getLogger(__name__)

BASELINE_PATH = DATA_DIR / "results_baseline.jsonl"
FINETUNED_PATH = DATA_DIR / "results_finetuned.jsonl"
ADVANCED_PATH = DATA_DIR / "results_advanced.jsonl"
DEBATE_PATH = DATA_DIR / "results_debate.jsonl"
DPO_PATH = DATA_DIR / "results_dpo.jsonl"
DPO_ADVANCED_PATH = DATA_DIR / "results_dpo_advanced.jsonl"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load JSONL records from disk."""

    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def validate_alignment(
    index: int,
    baseline_record: dict[str, str],
    finetuned_record: dict[str, str],
    advanced_record: dict[str, str],
    debate_record: dict[str, str],
    dpo_record: dict[str, str],
    dpo_advanced_record: dict[str, str],
) -> None:
    """Check that one row matches across all result files."""

    # Every row must refer to the same original sample.
    originals = {
        baseline_record["original_sentence"],
        finetuned_record["original_sentence"],
        advanced_record["original_slang"],
        debate_record["original_slang"],
        dpo_record["original_sentence"],
        dpo_advanced_record["original_slang"],
    }
    ground_truths = {
        baseline_record["ground_truth"],
        finetuned_record["ground_truth"],
        advanced_record["ground_truth"],
        debate_record["ground_truth"],
        dpo_record["ground_truth"],
        dpo_advanced_record["ground_truth"],
    }

    if len(originals) != 1 or len(ground_truths) != 1:
        raise ValueError(
            f"Row {index} is not aligned across all result files. "
            "Refusing to create a misleading merged dataset."
        )


def main() -> None:
    """Create merged JSONL and CSV files for evaluation."""

    configure_logging()
    # Load the six result files produced by different pipelines.
    baseline_records = load_jsonl(BASELINE_PATH)
    finetuned_records = load_jsonl(FINETUNED_PATH)
    advanced_records = load_jsonl(ADVANCED_PATH)
    debate_records = load_jsonl(DEBATE_PATH)
    dpo_records = load_jsonl(DPO_PATH)
    dpo_advanced_records = load_jsonl(DPO_ADVANCED_PATH)

    record_count = len(baseline_records)
    counts = {
        record_count,
        len(finetuned_records),
        len(advanced_records),
        len(debate_records),
        len(dpo_records),
        len(dpo_advanced_records),
    }
    if len(counts) != 1:
        raise ValueError(
            "The result files do not contain the same number of rows, so they "
            "cannot be safely merged."
        )

    merged_records: list[dict[str, str | int]] = []

    for index, (
        baseline_record,
        finetuned_record,
        advanced_record,
        debate_record,
        dpo_record,
        dpo_advanced_record,
    ) in enumerate(
        zip(
            baseline_records,
            finetuned_records,
            advanced_records,
            debate_records,
            dpo_records,
            dpo_advanced_records,
            strict=True,
        ),
        start=1,
    ):
        validate_alignment(
            index,
            baseline_record,
            finetuned_record,
            advanced_record,
            debate_record,
            dpo_record,
            dpo_advanced_record,
        )

        merged_records.append(
            {
                "id": index,
                "original_slang": baseline_record["original_sentence"],
                "ground_truth": baseline_record["ground_truth"],
                "baseline_output": baseline_record["deepseek_output"],
                "finetuned_output": finetuned_record["finetuned_output"],
                "advanced_output": advanced_record["final_translation"],
                "debate_output": debate_record["final_best_translation"],
                "dpo_output": dpo_record["finetuned_output"],
                "dpo_advanced_output": dpo_advanced_record["final_translation"],
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in merged_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Export CSV too for spreadsheet-based evaluation.
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(merged_records[0].keys()))
        writer.writeheader()
        writer.writerows(merged_records)

    logger.info("Merged %d rows into evaluation outputs", len(merged_records))
    print(f"Merged {len(merged_records)} rows into {OUTPUT_PATH}")
    print(f"Wrote CSV export to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
