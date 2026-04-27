# Data Directory

This directory is intentionally kept out of normal Git tracking.

It is used for:

- raw source data
- reconstructed TSV files
- processed JSONL files
- model result files
- merged evaluation files

## Why these files are not tracked

The course submission instructions ask that data should not be pushed to GitHub
as normal repository files. Large datasets and generated artifacts should be
stored with Git LFS or downloaded from an external source.

## Recommended setup

1. Download the OpenSub-Slang metadata package from the public benchmark repo.
2. Download the English OpenSubtitles corpus separately.
3. Reconstruct `slang_OpenSub.tsv` and `slang_OpenSub_negatives.tsv`.
4. Run the package preprocessing and experiment scripts.

## Public source used by this project

Benchmark repo:

- [amazon-science/slang-llm-benchmark](https://github.com/amazon-science/slang-llm-benchmark)

OpenSubtitles corpus:

- [OPUS OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php)

## Package commands after data is available

```bash
uv run slang-normalizer-preprocess
uv run slang-normalizer-prepare-mlx
uv run slang-normalizer-baseline
uv run slang-normalizer-finetuned
uv run slang-normalizer-advanced
uv run slang-normalizer-debate
uv run slang-normalizer-combine
```
