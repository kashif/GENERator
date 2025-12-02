#!/usr/bin/env python3
"""
Plot sequence recovery results for prokaryote and eukaryote datasets.
Generates figures matching the format of figures/sequence_recovery_*.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model name mapping for display
MODEL_DISPLAY_NAMES = {
    "GENERator-eukaryote-1.2b-base": "GENERator-1B",
    "GENERator-eukaryote-3b-base": "GENERator-3B",
    "GENERator-v2-eukaryote-1.2b-base": "GENERator-v2-1B",
    "GENERator-v2-eukaryote-3b-base": "GENERator-v2-3B",
    "GENERator-v2-prokaryote-1.2b-base": "GENERator-v2-1B",
    "GENERator-v2-prokaryote-3b-base": "GENERator-v2-3B",
    "evo-1-131k-base": "Evo-7B",
    "evo2_1b_base": "Evo2-1B",
    "evo2_7b_base": "Evo2-7B",
    "Qwen_Qwen3-4B-Base": "Qwen3-4B (LLM)",
    "Qwen_Qwen3-14B-Base": "Qwen3-14B (LLM)",
    "Qwen_Qwen3-30B-A3B-Base": "Qwen3-30B (LLM)",
}

def extract_model_name(filename):
    """Extract model name from parquet filename."""
    # Remove .parquet extension
    name = filename.replace(".parquet", "")

    # Remove _bfloat16 or _float32 suffix
    name = name.replace("_bfloat16", "").replace("_float32", "")

    # Map to display name
    for key, display_name in MODEL_DISPLAY_NAMES.items():
        if key in name:
            return display_name

    return name

def load_results(data_type):
    """Load all results for a given data type."""
    results_dir = Path("sequence_recovery_results") / data_type

    if not results_dir.exists():
        print(f"Warning: Directory {results_dir} does not exist")
        return {}

    results = {}
    for parquet_file in results_dir.glob("*.parquet"):
        model_name = extract_model_name(parquet_file.name)
        df = pd.read_parquet(parquet_file)

        # Calculate mean accuracy per organism type
        type_accuracy = df.groupby('type')['accuracy'].mean()
        results[model_name] = type_accuracy

    return results

def plot_sequence_recovery(data_type, output_path):
    """Generate sequence recovery bar plot for a given data type."""
    results = load_results(data_type)

    if not results:
        print(f"No results found for {data_type}")
        return

    # Combine all results into a DataFrame
    df_combined = pd.DataFrame(results)

    # Sort organism types alphabetically
    df_combined = df_combined.sort_index()

    # Define colors and patterns for models
    model_colors = {
        "Evo-7B": "#A8C5DD",
        "Evo2-1B": "#7BA7C7",
        "Evo2-7B": "#4F89B1",
        "GENERator-1B": "#90C5A9",
        "GENERator-3B": "#5A9A7A",
        "GENERator-v2-1B": "#7BA7C7",
        "GENERator-v2-3B": "#2E5C7F",
        "Qwen3-4B (LLM)": "#E8B4B8",
        "Qwen3-14B (LLM)": "#D88A8F",
        "Qwen3-30B (LLM)": "#C75E66",
    }

    # Hatching patterns
    model_patterns = {
        "Evo-7B": "",
        "Evo2-1B": "",
        "Evo2-7B": "",
        "GENERator-1B": "\\\\\\",
        "GENERator-3B": "\\\\\\",
        "GENERator-v2-1B": "///",
        "GENERator-v2-3B": "///",
        "Qwen3-4B (LLM)": "...",
        "Qwen3-14B (LLM)": "...",
        "Qwen3-30B (LLM)": "...",
    }

    # Sort models in desired order (LLM baselines at the end)
    model_order = ["Evo-7B", "Evo2-1B", "Evo2-7B", "GENERator-1B", "GENERator-3B", "GENERator-v2-1B", "GENERator-v2-3B",
                    "Qwen3-4B (LLM)", "Qwen3-14B (LLM)", "Qwen3-30B (LLM)"]
    available_models = [m for m in model_order if m in df_combined.columns]
    df_combined = df_combined[available_models]

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))

    # Bar width and positions
    n_models = len(df_combined.columns)
    n_types = len(df_combined.index)
    bar_width = 0.15
    x = np.arange(n_types)

    # Plot bars for each model
    for i, model in enumerate(df_combined.columns):
        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            df_combined[model],
            bar_width,
            label=model,
            color=model_colors.get(model, "#888888"),
            hatch=model_patterns.get(model, ""),
            edgecolor='white',
            linewidth=0.5
        )

        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Customize plot
    title = f"Sequence Recovery Accuracy ({'Prokaryote' if data_type == 'bacteria' else 'Eukaryote'})"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_combined.index, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    """Generate both prokaryote and eukaryote plots."""
    os.makedirs("figures", exist_ok=True)

    print("Generating sequence recovery plots...")

    # Generate prokaryote plot - save to separate file to preserve originals
    plot_sequence_recovery("bacteria", "figures/sequence_recovery_prokaryote_reproduced.png")

    # Generate eukaryote plot - save to separate file to preserve originals
    plot_sequence_recovery("eukaryote", "figures/sequence_recovery_eukaryote_reproduced.png")

    print("Done!")
    print("\nNote: Original figures preserved. Our results saved as:")
    print("  - figures/sequence_recovery_prokaryote_reproduced.png")
    print("  - figures/sequence_recovery_eukaryote_reproduced.png")

if __name__ == "__main__":
    main()
