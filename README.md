# Slang Normalization Final Project

This repository contains our final project for slang normalization.

The goal of this project is to rewrite slang-heavy English sentences into
clearer and more formal English while keeping the original meaning.

This repository has two levels:

- the **repository root**, which stores course-level materials
- the **`slang_normalizer/`** folder, which contains the Python package itself

If you want to understand the code, start with:

- [slang_normalizer/README.md](./slang_normalizer/README.md)

## What Is Included

- the Python package source code
- experiment scripts for preprocessing, training, inference, and evaluation
- test files
- environment setup files
- architecture documentation
- proposal and reference materials

## Main Package Location

The actual Python project lives in:

- [slang_normalizer/](./slang_normalizer/)

That folder contains:

- `pyproject.toml`
- `src/slang_normalizer/`
- `tests/`
- `docs/`
- the package README

## Data Note

Large data files, generated results, and model checkpoints are not kept in the
public Git tracking history for submission purposes.

The local `data/` folder may still exist on the machine for running
experiments, but the repository only keeps documentation about those files.

See:

- [data/README.md](./data/README.md)
- [slang_normalizer/data/README.md](./slang_normalizer/data/README.md)

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate slang-normalizer
```

Or use `uv` inside the package folder:

```bash
cd slang_normalizer
uv sync
```

Then read the full package guide here:

- [slang_normalizer/README.md](./slang_normalizer/README.md)

## Repository Layout

```text
5400FinalProject/
├── README.md
├── environment.yml
├── Project Proposal.docx
├── Project Proposal.pdf
├── data/
│   └── README.md
├── references/
├── slang_normalizer/
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── docs/
│   │   ├── architecture.svg
│   │   └── architecture.drawio
│   ├── src/
│   │   └── slang_normalizer/
│   ├── tests/
│   └── data/
│       └── README.md
```

## Submission Note

This root README is a project-level guide.

The detailed software documentation, package commands, architecture diagram,
and experiment workflow are documented in:

- [slang_normalizer/README.md](./slang_normalizer/README.md)
