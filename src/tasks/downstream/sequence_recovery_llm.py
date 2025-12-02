#!/usr/bin/env python3
"""
Sequence recovery evaluation for standard LLMs using vLLM.
Evaluates how well general-purpose language models can predict DNA sequences.
"""

import os
import argparse
import time
from typing import List
from tqdm import tqdm
import pandas as pd
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="LLM sequence recovery evaluation using vLLM")
    parser.add_argument(
        "--data_type",
        default="eukaryote",
        choices=["eukaryote", "bacteria", "others"],
        help="Data type (eukaryote, bacteria, or others)."
    )
    parser.add_argument(
        "--data_path",
        default="hf://datasets/GenerTeam/sequence-recovery",
        help="Download from https://huggingface.co/datasets/GenerTeam/sequence-recovery"
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-3B",
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-3B, meta-llama/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--output_dir",
        default="./sequence_recovery_results",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum input sequence length (truncate left, keep rightmost)."
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=30,
        help="Number of characters to generate (30 nucleotides)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for vLLM inference."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing). None = all samples."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism."
    )
    return parser.parse_args()

def calculate_accuracy(predictions: List[str], labels: List[str], seq_length: int = 30) -> List[float]:
    """Calculate accuracy for each sequence.

    Note: Labels may be longer than seq_length, but we only compare the first seq_length characters.
    """
    accuracies = []
    for label, pred in zip(labels, predictions):
        # Only compare first seq_length characters of label with prediction
        label_truncated = label[:seq_length]
        # Clean prediction - remove any non-ACGT characters and take first seq_length
        pred_clean = ''.join(c.upper() for c in pred if c.upper() in 'ACGT')[:seq_length]

        same_count = sum(1 for i in range(min(len(label_truncated), len(pred_clean), seq_length))
                        if label_truncated[i] == pred_clean[i])
        accuracies.append(same_count / seq_length)
    return accuracies

def process_checkpoint(args: argparse.Namespace) -> None:
    """Process a single checkpoint and return results."""
    print("\n" + "=" * 80)
    print("🧬  DNA SEQUENCE RECOVERY PIPELINE (LLM + vLLM)  🧬")
    print("=" * 80 + "\n")
    print(f"Processing model {args.model_name}...")

    # Load data
    print("Loading data...")
    data_path = f"{args.data_path}/{args.data_type}/test.parquet"
    df = pd.read_parquet(data_path)

    # Limit samples if specified (for testing)
    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Limited to {args.max_samples} samples for testing")

    total_sequences = len(df)
    print(f"Total sequences to process: {total_sequences}")

    # Import vLLM
    try:
        from vllm import LLM, SamplingParams
        print("vLLM imported successfully")
    except ImportError:
        print("ERROR: vLLM not installed. Install with: pip install vllm")
        return

    # Load LLM with vLLM
    print(f"Loading LLM: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_seq_len + args.gen_len,  # Input + generation
        trust_remote_code=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,  # Avoid compatibility issues
        enforce_eager=True,  # Disable CUDA graph to avoid deep_gemm issues
    )

    # Get tokenizer to check for special tokens
    tokenizer = llm.get_tokenizer()

    # Set up sampling parameters for greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        top_p=1.0,
        max_tokens=args.gen_len,
        stop=[],  # No stop sequences - we want exactly gen_len tokens
    )

    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    types = df['type'].tolist()

    predictions = []

    print(f"Processing {total_sequences} sequences with batch_size={args.batch_size}...")
    start_time = time.time()

    # Prepare prompts
    prompts = []
    for seq in sequences:
        # Truncate sequence to max length
        truncated_seq = seq[-min(len(seq), args.max_seq_len):]

        # Create a simple prompt that asks the model to continue the DNA sequence
        # We don't add any special formatting - just the raw DNA sequence
        prompt = truncated_seq
        prompts.append(prompt)

    # Debug: Show first prompt
    print(f"\nDebug - First prompt (last 100 chars):")
    print(f"  ...{prompts[0][-100:]}")
    print(f"  Prompt length: {len(prompts[0])} characters")

    # Generate predictions using vLLM (batched)
    print("\nGenerating predictions...")
    with tqdm(total=total_sequences, desc="Inference", unit="seq") as pbar:
        for i in range(0, total_sequences, args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]

            # Generate with vLLM
            outputs = llm.generate(batch_prompts, sampling_params)

            # Extract predictions
            for output in outputs:
                generated_text = output.outputs[0].text
                # Clean up the generated text - extract only ACGT characters
                pred = ''.join(c.upper() for c in generated_text if c.upper() in 'ACGT')[:args.gen_len]
                predictions.append(pred)

            # Debug: Show first batch predictions
            if i == 0:
                print(f"\nDebug - First batch predictions:")
                for j, output in enumerate(outputs[:3]):
                    raw_text = output.outputs[0].text
                    clean_pred = predictions[j]
                    print(f"  Sample {j}:")
                    print(f"    Raw output (first 50 chars): {raw_text[:50]}")
                    print(f"    Cleaned prediction: {clean_pred}")
                    print(f"    Label (first 30): {labels[i+j][:30]}")
                    print()

            pbar.update(len(batch_prompts))

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")
    print(f"Speed: {total_sequences/elapsed_time:.2f} sequences/second")

    # Create results dataframe
    results_df = pd.DataFrame({
        'pred': predictions,
        'label': labels,
        'type': types
    })

    # Calculate accuracy
    accuracies = calculate_accuracy(predictions, labels)
    results_df['accuracy'] = accuracies

    # Calculate statistics
    type_means = results_df.groupby('type')['accuracy'].mean()
    overall_mean = results_df['accuracy'].mean()

    # Save detailed results to parquet
    os.makedirs(f"{args.output_dir}/{args.data_type}", exist_ok=True)

    # Extract model name for filename (replace / with _)
    model_name_safe = args.model_name.replace("/", "_")
    output_filename = f"{model_name_safe}.parquet"
    output_path = os.path.join(args.output_dir, args.data_type, output_filename)
    results_df[['pred', 'label', 'type', 'accuracy']].to_parquet(output_path)

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Overall Accuracy: {overall_mean:.4f} ({overall_mean*100:.2f}%)")
    print(f"\nType-wise Accuracy:")
    for type_name, acc in type_means.items():
        print(f"  {type_name}: {acc:.4f} ({acc*100:.2f}%)")
    print("-" * 80)
    print(f"✅ Completed {args.model_name}")
    print(f"📊 Results saved to: {output_path}")
    print("-" * 80)

def main():
    args = parse_args()
    process_checkpoint(args)

if __name__ == "__main__":
    main()
