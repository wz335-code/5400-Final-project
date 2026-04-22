from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import save_config

from slang_normalizer.train_mlx import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    ensure_model_access,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "dpo_train.jsonl"
DEFAULT_VALID_PATH = REPO_ROOT / "data" / "dpo_valid.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "slang_normalizer" / "checkpoints_dpo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPO LoRA adapter on Apple MLX using preference data."
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
        help="Path to the starting SFT adapter directory.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the DPO training JSONL file.",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=DEFAULT_VALID_PATH,
        help="Path to the DPO validation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the DPO adapter checkpoints should be saved.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter controlling preference sharpness.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DPO training.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of DPO training iterations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for the DPO optimizer.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save adapter checkpoints every N iterations.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Evaluate on the validation set every N iterations.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for prompt+response pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--logp-reduction",
        choices=("sum", "mean"),
        default="mean",
        help=(
            "How to aggregate token log-probabilities over the completion. "
            "'mean' is usually safer when chosen and rejected responses have "
            "different lengths."
        ),
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.2,
        help=(
            "Weight for an auxiliary chosen-response likelihood loss that keeps "
            "the policy close to fluent supervised behavior."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def load_adapter_config(adapter_path: Path) -> dict:
    config_path = adapter_path / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def ensure_eos(tokens: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return tokens
    if not tokens or tokens[-1] != eos_token_id:
        return [*tokens, eos_token_id]
    return tokens


def encode_dataset(
    records: list[dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> list[dict[str, object]]:
    encoded_records: list[dict[str, object]] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for record in records:
        prompt = record["prompt"]
        chosen = record["chosen"]
        rejected = record["rejected"]

        prompt_tokens = tokenizer.encode(prompt)
        chosen_tokens = ensure_eos(tokenizer.encode(prompt + chosen), eos_token_id)
        rejected_tokens = ensure_eos(tokenizer.encode(prompt + rejected), eos_token_id)
        prompt_length = len(prompt_tokens)

        if len(chosen_tokens) > max_seq_length or len(rejected_tokens) > max_seq_length:
            continue
        if prompt_length >= len(chosen_tokens) or prompt_length >= len(rejected_tokens):
            continue

        encoded_records.append(
            {
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "chosen_prompt_length": prompt_length,
                "rejected_prompt_length": prompt_length,
            }
        )

    return encoded_records


def pad_batch(
    examples: list[dict[str, object]],
    key: str,
    prompt_key: str,
) -> tuple[mx.array, mx.array, mx.array]:
    lengths = [len(example[key]) for example in examples]
    prompt_lengths = [int(example[prompt_key]) for example in examples]
    max_length = max(lengths)
    batch = np.zeros((len(examples), max_length), dtype=np.int32)

    for index, example in enumerate(examples):
        tokens = example[key]
        batch[index, : len(tokens)] = tokens

    return (
        mx.array(batch),
        mx.array(prompt_lengths),
        mx.array(lengths),
    )


def create_batches(
    records: list[dict[str, object]],
    batch_size: int,
    seed: int,
):
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch_examples = [records[i] for i in indices[start : start + batch_size]]
        chosen_batch, chosen_prompt_lengths, chosen_lengths = pad_batch(
            batch_examples,
            key="chosen_tokens",
            prompt_key="chosen_prompt_length",
        )
        rejected_batch, rejected_prompt_lengths, rejected_lengths = pad_batch(
            batch_examples,
            key="rejected_tokens",
            prompt_key="rejected_prompt_length",
        )

        yield (
            chosen_batch,
            chosen_prompt_lengths,
            chosen_lengths,
            rejected_batch,
            rejected_prompt_lengths,
            rejected_lengths,
        )


def sequence_logps(
    model,
    batch: mx.array,
    prompt_lengths: mx.array,
    lengths: mx.array,
    reduction: str,
) -> mx.array:
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    token_log_probs = mx.take_along_axis(log_probs, targets[..., None], axis=-1)
    token_log_probs = token_log_probs.squeeze(-1)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(
        steps >= prompt_lengths[:, None],
        steps < lengths[:, None],
    )
    masked_log_probs = token_log_probs * mask
    completion_logps = masked_log_probs.sum(axis=1)

    if reduction == "mean":
        token_counts = mx.maximum(mask.sum(axis=1), 1)
        return completion_logps / token_counts

    return completion_logps


def precompute_reference_logps(
    model,
    records: list[dict[str, object]],
    batch_size: int,
    seed: int,
    reduction: str,
) -> list[dict[str, float]]:
    reference_stats: list[dict[str, float]] = [None] * len(records)  # type: ignore

    indexed = list(enumerate(records))
    random.Random(seed).shuffle(indexed)

    model.eval()

    for start in range(0, len(indexed), batch_size):
        chunk = indexed[start : start + batch_size]
        batch_examples = [record for _, record in chunk]

        chosen_batch, chosen_prompt_lengths, chosen_lengths = pad_batch(
            batch_examples,
            key="chosen_tokens",
            prompt_key="chosen_prompt_length",
        )
        rejected_batch, rejected_prompt_lengths, rejected_lengths = pad_batch(
            batch_examples,
            key="rejected_tokens",
            prompt_key="rejected_prompt_length",
        )

        chosen_logps = sequence_logps(
            model,
            chosen_batch,
            chosen_prompt_lengths,
            chosen_lengths,
            reduction,
        )
        rejected_logps = sequence_logps(
            model,
            rejected_batch,
            rejected_prompt_lengths,
            rejected_lengths,
            reduction,
        )
        mx.eval(chosen_logps, rejected_logps)

        for (index, _), chosen_logp, rejected_logp in zip(
            chunk,
            chosen_logps.tolist(),
            rejected_logps.tolist(),
            strict=True,
        ):
            reference_stats[index] = {
                "ref_chosen_logp": float(chosen_logp),
                "ref_rejected_logp": float(rejected_logp),
            }

    return reference_stats


def attach_reference_logps(
    records: list[dict[str, object]],
    reference_stats: list[dict[str, float]],
) -> list[dict[str, object]]:
    attached: list[dict[str, object]] = []

    for record, stats in zip(records, reference_stats, strict=True):
        attached.append(
            {
                **record,
                "ref_chosen_logp": stats["ref_chosen_logp"],
                "ref_rejected_logp": stats["ref_rejected_logp"],
            }
        )

    return attached


def dpo_loss(model, batch, beta: float, reduction: str):
    (
        chosen_batch,
        chosen_prompt_lengths,
        chosen_lengths,
        rejected_batch,
        rejected_prompt_lengths,
        rejected_lengths,
        ref_chosen_logps,
        ref_rejected_logps,
    ) = batch

    policy_chosen_logps = sequence_logps(
        model,
        chosen_batch,
        chosen_prompt_lengths,
        chosen_lengths,
        reduction,
    )
    policy_rejected_logps = sequence_logps(
        model,
        rejected_batch,
        rejected_prompt_lengths,
        rejected_lengths,
        reduction,
    )

    logits = beta * (
        (policy_chosen_logps - policy_rejected_logps)
        - (ref_chosen_logps - ref_rejected_logps)
    )
    losses = -nn.log_sigmoid(logits)
    accuracy = mx.mean(logits > 0)
    return losses.mean(), accuracy


def chosen_anchor_loss(
    model,
    batch,
) -> mx.array:
    (
        chosen_batch,
        chosen_prompt_lengths,
        chosen_lengths,
        _rejected_batch,
        _rejected_prompt_lengths,
        _rejected_lengths,
        _ref_chosen_logps,
        _ref_rejected_logps,
    ) = batch

    chosen_mean_logps = sequence_logps(
        model,
        chosen_batch,
        chosen_prompt_lengths,
        chosen_lengths,
        reduction="mean",
    )
    return -chosen_mean_logps.mean()


def total_training_loss(
    model,
    batch,
    beta: float,
    reduction: str,
    anchor_weight: float,
) -> mx.array:
    preference_loss, _ = dpo_loss(model, batch, beta, reduction)
    if anchor_weight <= 0:
        return preference_loss
    return preference_loss + (anchor_weight * chosen_anchor_loss(model, batch))


def build_training_batch(batch_examples: list[dict[str, object]]):
    chosen_batch, chosen_prompt_lengths, chosen_lengths = pad_batch(
        batch_examples,
        key="chosen_tokens",
        prompt_key="chosen_prompt_length",
    )
    rejected_batch, rejected_prompt_lengths, rejected_lengths = pad_batch(
        batch_examples,
        key="rejected_tokens",
        prompt_key="rejected_prompt_length",
    )
    ref_chosen_logps = mx.array(
        [example["ref_chosen_logp"] for example in batch_examples],
        dtype=mx.float32,
    )
    ref_rejected_logps = mx.array(
        [example["ref_rejected_logp"] for example in batch_examples],
        dtype=mx.float32,
    )

    return (
        chosen_batch,
        chosen_prompt_lengths,
        chosen_lengths,
        rejected_batch,
        rejected_prompt_lengths,
        rejected_lengths,
        ref_chosen_logps,
        ref_rejected_logps,
    )


def iterate_training_batches(
    records: list[dict[str, object]],
    batch_size: int,
    seed: int,
):
    step_seed = seed
    while True:
        indices = list(range(len(records)))
        random.Random(step_seed).shuffle(indices)
        step_seed += 1
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            yield build_training_batch(
                [records[index] for index in indices[start : start + batch_size]]
            )


def evaluate_dpo(
    model,
    records: list[dict[str, object]],
    batch_size: int,
    beta: float,
    reduction: str,
) -> tuple[float, float]:
    if not records:
        return math.nan, math.nan

    model.eval()
    losses = []
    accuracies = []

    for start in range(0, len(records), batch_size):
        batch_examples = records[start : start + batch_size]
        batch = build_training_batch(batch_examples)
        loss_value, accuracy = dpo_loss(model, batch, beta, reduction)
        mx.eval(loss_value, accuracy)
        losses.append(float(loss_value.item()))
        accuracies.append(float(accuracy.item()))

    model.train()
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def save_adapter_weights(model, output_dir: Path, step: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))

    latest_path = output_dir / "adapters.safetensors"
    mx.save_safetensors(str(latest_path), adapter_weights)

    if step is not None:
        checkpoint_path = output_dir / f"{step:07d}_adapters.safetensors"
        mx.save_safetensors(str(checkpoint_path), adapter_weights)


def main() -> None:
    args = parse_args()
    ensure_model_access(args.model)

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    train_records = load_jsonl(args.train_path)
    valid_records = load_jsonl(args.valid_path)

    print("Loading base model and tokenizer")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True},
    )

    adapter_config = load_adapter_config(args.adapter_path)
    model.freeze()
    linear_to_lora_layers(
        model,
        adapter_config["num_layers"],
        adapter_config["lora_parameters"],
        use_dora=(adapter_config.get("fine_tune_type") == "dora"),
    )
    model.load_weights(str(args.adapter_path / "adapters.safetensors"), strict=False)
    print_trainable_parameters(model)

    print("Encoding DPO datasets")
    encoded_train = encode_dataset(train_records, tokenizer, args.max_seq_length)
    encoded_valid = encode_dataset(valid_records, tokenizer, args.max_seq_length)

    if len(encoded_train) < args.batch_size:
        raise ValueError(
            f"Need at least batch_size={args.batch_size} DPO examples, but only "
            f"have {len(encoded_train)} usable train pairs."
        )

    print("Precomputing reference log-probabilities from the starting SFT adapter")
    reference_train = precompute_reference_logps(
        model,
        encoded_train,
        batch_size=args.batch_size,
        seed=args.seed,
        reduction=args.logp_reduction,
    )
    reference_valid = precompute_reference_logps(
        model,
        encoded_valid,
        batch_size=max(1, min(args.batch_size, len(encoded_valid))),
        seed=args.seed,
        reduction=args.logp_reduction,
    )

    train_data = attach_reference_logps(encoded_train, reference_train)
    valid_data = attach_reference_logps(encoded_valid, reference_valid)
    model.train()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(
        {
            "adapter_path": str(output_dir),
            "base_model": args.model,
            "source_sft_adapter_path": str(args.adapter_path),
            "train_path": str(args.train_path),
            "valid_path": str(args.valid_path),
            "beta": args.beta,
            "batch_size": args.batch_size,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
            "save_every": args.save_every,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
            "logp_reduction": args.logp_reduction,
            "anchor_weight": args.anchor_weight,
            "fine_tune_type": adapter_config.get("fine_tune_type", "lora"),
            "num_layers": adapter_config["num_layers"],
            "lora_parameters": {
                **adapter_config["lora_parameters"],
                "rank": 16,
            },
            "dpo": True,
        },
        output_dir / "adapter_config.json",
    )

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_value_and_grad = nn.value_and_grad(
        model,
        lambda current_model, batch: total_training_loss(
            current_model,
            batch,
            args.beta,
            args.logp_reduction,
            args.anchor_weight,
        ),
    )

    batch_iterator = iterate_training_batches(
        train_data,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    best_valid_loss = math.inf
    print(f"Starting DPO training for {args.iters} iterations")

    for step in range(1, args.iters + 1):
        batch = next(batch_iterator)
        loss_value, grads = loss_value_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss_value)

        if step % 10 == 0 or step == 1:
            total_loss = total_training_loss(
                model,
                batch,
                args.beta,
                args.logp_reduction,
                args.anchor_weight,
            )
            train_loss, train_accuracy = dpo_loss(
                model,
                batch,
                args.beta,
                args.logp_reduction,
            )
            anchor_loss = chosen_anchor_loss(model, batch)
            mx.eval(total_loss, train_loss, train_accuracy, anchor_loss)
            print(
                f"Iter {step}: total_loss={total_loss.item():.4f}, "
                f"dpo_loss={train_loss.item():.4f}, "
                f"anchor_loss={anchor_loss.item():.4f}, "
                f"train_pref_acc={train_accuracy.item():.4f}"
            )

        if step % args.eval_every == 0 or step == args.iters:
            valid_loss, valid_accuracy = evaluate_dpo(
                model,
                valid_data,
                batch_size=max(1, min(args.batch_size, len(valid_data))),
                beta=args.beta,
                reduction=args.logp_reduction,
            )
            print(
                f"Iter {step}: valid_loss={valid_loss:.4f}, "
                f"valid_pref_acc={valid_accuracy:.4f}"
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_adapter_weights(model, output_dir)
                print(
                    "Saved improved DPO adapter weights to "
                    f"{output_dir / 'adapters.safetensors'}"
                )

        if step % args.save_every == 0:
            save_adapter_weights(model, output_dir, step=step)
            print(f"Saved checkpoint at step {step}")

    final_path = output_dir / "final_adapters.safetensors"
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(final_path), adapter_weights)
    print(f"Saved final-step DPO adapter weights to {final_path}")


if __name__ == "__main__":
    main()
