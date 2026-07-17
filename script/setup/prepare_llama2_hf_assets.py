"""Prepare local Hugging Face config/tokenizer assets from an official Meta Llama 2 download.

MemoryVLA inference loads all trained weights from its own checkpoint, but the
Prismatic LLM backbone still needs a Hugging Face-compatible LlamaConfig and
tokenizer. This script converts Meta's native ``params.json`` and
``tokenizer.model`` without copying or converting the base Llama weights.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaTokenizer


def calculate_intermediate_size(params: dict, hidden_size: int) -> int:
    """Reproduce Meta Llama's feed-forward dimension calculation."""
    intermediate_size = int(2 * (4 * hidden_size) / 3)
    if params.get("ffn_dim_multiplier") is not None:
        intermediate_size = int(params["ffn_dim_multiplier"] * intermediate_size)

    multiple_of = int(params.get("multiple_of", 256))
    return multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing Meta's params.json and tokenizer.model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for the local Hugging Face-compatible assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params_path = args.meta_checkpoint_dir / "params.json"
    tokenizer_path = args.meta_checkpoint_dir / "tokenizer.model"
    if not params_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Expected params.json and tokenizer.model under {args.meta_checkpoint_dir}"
        )

    params = json.loads(params_path.read_text(encoding="utf-8"))
    hidden_size = int(params["dim"])
    num_attention_heads = int(params["n_heads"])
    vocab_size = int(params.get("vocab_size", -1))
    if vocab_size not in {-1, 32000}:
        raise ValueError(f"Expected the Llama 2 vocabulary size, got {vocab_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=hidden_size,
        intermediate_size=calculate_intermediate_size(params, hidden_size),
        num_hidden_layers=int(params["n_layers"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(params.get("n_kv_heads") or num_attention_heads),
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=float(params.get("norm_eps", 1e-5)),
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=float(params.get("rope_theta", 10000.0)),
        torch_dtype="float16",
    )
    config.save_pretrained(args.output_dir)

    tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_path),
        legacy=True,
        model_max_length=2048,
        add_bos_token=True,
        add_eos_token=False,
    )
    tokenizer.save_pretrained(args.output_dir)

    loaded_config = AutoConfig.from_pretrained(args.output_dir, local_files_only=True)
    loaded_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True)
    if loaded_config.model_type != "llama" or len(loaded_tokenizer) != 32000:
        raise RuntimeError("Generated assets failed local Hugging Face validation")

    print(f"Prepared local Llama 2 assets in: {args.output_dir}")
    print(f"Config: {loaded_config.model_type}, hidden_size={loaded_config.hidden_size}")
    print(f"Tokenizer: {type(loaded_tokenizer).__name__}, vocab_size={len(loaded_tokenizer)}")
    print(f"export MEMVLA_LLAMA2_7B_PATH={args.output_dir}")


if __name__ == "__main__":
    main()
