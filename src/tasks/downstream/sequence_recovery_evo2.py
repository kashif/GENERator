import os
import argparse
import hashlib
from tqdm import tqdm
import pandas as pd
import torch
from evo2 import Evo2
import time
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description="Evo2 batch inference on Parquet")
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
        default="evo2_7b",
        choices=["evo2_1b", "evo2_7b", "evo2_40b", "evo2_1b_base", "evo2_7b_base", "evo2_40b_base"],
        help="Evo2 model variant to use",
    )
    parser.add_argument(
        "--output_dir",
        default="./sequence_recovery_results",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum input sequence length (truncate left, keep rightmost). Reduced from 6144 to avoid OOM with Evo2's FFT-based architecture."
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=30,
        help="Number of tokens to generate beyond the prompt. Evo2 uses character-level tokenization (1 token = 1 nucleotide)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference. Evo2 supports batched generation with Transformer Engine and Flash Attention."
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
        same_count = sum(1 for i in range(min(len(label_truncated), len(pred), seq_length))
                        if label_truncated[i] == pred[i])
        accuracies.append(same_count / seq_length)
    return accuracies

def process_checkpoint(args: argparse.Namespace) -> None:
    """Process a single checkpoint and return results."""
    print("\n" + "=" * 80)
    print("🧬  DNA SEQUENCE RECOVERY PIPELINE (EVO2)  🧬")
    print("=" * 80 + "\n")
    print(f"Processing checkpoint {args.model_name}...")

    # Load data
    print("Loading data...")
    data_path = f"{args.data_path}/{args.data_type}/test.parquet"
    df = pd.read_parquet(data_path)
    total_sequences = len(df)

    print(f"Loading Evo2 model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Evo2 model
    evo2_model = Evo2(args.model_name)

    # Adjust batch size for larger models to avoid OOM
    if '7b' in args.model_name or '40b' in args.model_name:
        args.batch_size = min(args.batch_size, 8)
        print(f"Reduced batch size to {args.batch_size} for {args.model_name} to avoid OOM")

    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    types = df['type'].tolist()

    predictions = []

    print(f"Processing {total_sequences} sequences...")
    start_time = time.time()

    with tqdm(total=total_sequences, desc="Inference", unit="seq") as pbar:
        for i in range(0, total_sequences, args.batch_size):
            batch_seqs = sequences[i:i + args.batch_size]

            # Truncate sequences to max length
            truncated_seqs = [
                seq[-min(len(seq), args.max_seq_len):]
                for seq in batch_seqs
            ]

            # Generate using Evo2 with optimizations enabled
            # - batched=True: Enables batch processing (default, uses Transformer Engine)
            # - cached_generation=True: Enables KV caching for faster generation (default)
            # - Flash Attention: Automatically used if installed (required dependency)
            try:
                output = evo2_model.generate(
                    prompt_seqs=truncated_seqs,
                    n_tokens=args.gen_len,
                    temperature=0.0,  # Greedy decoding for deterministic results
                    top_k=1,
                    batched=True,  # Explicitly enable batched generation
                    cached_generation=True,  # Explicitly enable KV caching
                    verbose=0,  # Disable verbose output for cleaner logs
                )

                # Evo2 returns GenerationOutput with .sequences attribute containing only generated text
                # output.sequences is a list of strings with the generated nucleotides
                batch_preds = output.sequences

                # Debug: Print first batch to verify
                if i == 0:
                    print(f"\nDebug - First batch:")
                    print(f"  Number of predictions: {len(batch_preds)}")
                    print(f"  First prediction length: {len(batch_preds[0])}")
                    print(f"  First prediction: {batch_preds[0][:50]}...")

                predictions.extend(batch_preds)

            except Exception as e:
                print(f"\nError processing batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # Add empty predictions for failed batches
                predictions.extend([""] * len(batch_seqs))

            pbar.update(len(batch_seqs))

    elapsed_time = time.time() - start_time
    print(f"Inference completed in {elapsed_time:.2f} seconds")

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
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.model_name}.parquet"
    output_path = os.path.join(args.output_dir, output_filename)
    results_df[['pred', 'label', 'type', 'accuracy']].to_parquet(output_path)

    print(f"Model {args.model_name} - Overall Accuracy: {overall_mean:.4f}")
    print(f"Model {args.model_name} - Type-wise Accuracy:\n{type_means}")
    print("-" * 80)
    print(f"✅ Completed {args.model_name}")
    print(f"📊 Results saved to: {output_path}")
    print("-" * 80)

def main():
    args = parse_args()
    process_checkpoint(args)

if __name__ == "__main__":
    main()
