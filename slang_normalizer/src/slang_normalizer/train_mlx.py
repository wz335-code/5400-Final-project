"""Write and optionally run the MLX LoRA training configuration."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mlx"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "slang_normalizer" / "checkpoints"
DEFAULT_CONFIG_PATH = DEFAULT_CHECKPOINT_DIR / "mlx_lora_config.yaml"
DEFAULT_ADAPTER_PATH = DEFAULT_CHECKPOINT_DIR / "llama3_slang_lora"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# Create the MLX config and optionally launch LoRA training.
logger = logging.getLogger(__name__)


def build_training_config(
    data_dir: Path,
    adapter_path: Path,
    iters: int,
    batch_size: int,
    num_layers: int,
    lora_rank: int,
    model: str,
) -> dict[str, object]:
    """Build the MLX LoRA training configuration dictionary."""

    # Keep the main training settings in one place.
    return {
        "model": model,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "seed": 42,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "iters": iters,
        "val_batches": 25,
        "learning_rate": 1e-5,
        "steps_per_report": 10,
        "steps_per_eval": 100,
        "save_every": 100,
        "adapter_path": str(adapter_path),
        "max_seq_length": 2048,
        "grad_checkpoint": True,
        "lora_parameters": {
            "rank": lora_rank,
            "dropout": 0.0,
            "scale": 20.0,
        },
    }


def write_config(config: dict[str, object], config_path: Path) -> None:
    """Write the MLX config file to disk."""

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=2)
        output_file.write("\n")


def build_command(config_path: Path) -> list[str]:
    """Build the command used to launch MLX LoRA training."""

    return ["mlx_lm.lora", "--config", str(config_path)]


def ensure_model_access(model: str) -> None:
    """Check that the selected Hugging Face model can be accessed."""

    # Check gated-model access before starting a long training run.
    if Path(model).exists():
        return

    api = HfApi()

    try:
        api.model_info(model)
    except GatedRepoError as error:
        raise RuntimeError(
            "Cannot access the gated Hugging Face model "
            f"'{model}'.\n\n"
            "Before training, do these two steps:\n"
            "1. Open the model page in your browser and request/accept access:\n"
            f"   https://huggingface.co/{model}\n"
            "2. Log in from this machine with a personal Hugging Face token:\n"
            "   hf auth login\n\n"
            "After that, rerun the same training command."
        ) from error
    except RepositoryNotFoundError as error:
        raise RuntimeError(
            f"The model '{model}' was not found on Hugging Face. "
            "Check the repo name or pass a local converted model path."
        ) from error


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for MLX training."""

    parser = argparse.ArgumentParser(
        description="Create the MLX LoRA training config and optionally run training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face repo id or local path for the MLX-compatible base model.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.jsonl and valid.jsonl for MLX.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Where to write the MLX LoRA config file.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Directory where MLX should save adapter checkpoints.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-step batch size for MLX LoRA training.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="Number of transformer layers to fine-tune with LoRA.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank to use in mlx_lm.lora_parameters.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch mlx_lm.lora immediately after writing the config.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the training config and optionally launch training."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    config = build_training_config(
        data_dir=args.data_dir,
        adapter_path=args.adapter_path,
        iters=args.iters,
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        lora_rank=args.lora_rank,
        model=args.model,
    )
    write_config(config, args.config_path)

    command = build_command(args.config_path)
    printable_command = " ".join(shlex.quote(part) for part in command)

    print(f"Wrote MLX LoRA config to {args.config_path}")
    print(f"Adapter checkpoints will be saved under {args.adapter_path}")
    print("Run training with:")
    print(f"uv run {printable_command}")
    logger.info("Training config written to %s", args.config_path)

    if args.run:
        # Run the MLX command only after the config file is ready.
        ensure_model_access(args.model)
        logger.info("Starting MLX LoRA training")
        subprocess.run(command, cwd=REPO_ROOT / "slang_normalizer", check=True)


if __name__ == "__main__":
    main()
