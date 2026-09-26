"""
visualization.py
=================
Generates comparison charts strictly from the actual benchmark_results.csv
produced by benchmark.py. Never manually enter or invent values here — if
the CSV doesn't exist yet, this script tells you to run the benchmark first
instead of drawing placeholder data.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def _load_dataframe():
    import pandas as pd

    if not os.path.exists(config.RESULTS_CSV):
        raise FileNotFoundError(
            f"{config.RESULTS_CSV} not found. Run the benchmark first:\n"
            f"    python run_benchmark.py\n"
        )
    df = pd.read_csv(config.RESULTS_CSV)
    if df.empty:
        raise ValueError(
            f"{config.RESULTS_CSV} is empty. Re-run the benchmark — no "
            f"model produced usable results."
        )
    return df


def _aggregate(df):
    return df.groupby("model").agg(
        mean_wer=("wer", "mean"),
        mean_inference_time=("inference_time", "mean"),
        mean_rtf=("rtf", "mean"),
        mean_memory_mb=("memory_usage_mb", "mean"),
    ).reset_index()


def generate_all_charts():
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    df = _load_dataframe()
    agg = _aggregate(df)
    models = agg["model"].tolist()

    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None

    # 1. WER comparison bar chart
    plt.figure(figsize=(7, 5))
    plt.bar(models, agg["mean_wer"], color=["#4C72B0", "#55A868", "#C44E52"][:len(models)])
    plt.ylabel("Mean Word Error Rate (lower is better)")
    plt.title("WER Comparison Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "wer_comparison.png"), dpi=150)
    plt.close()

    # 2. Inference time comparison
    plt.figure(figsize=(7, 5))
    plt.bar(models, agg["mean_inference_time"], color=["#4C72B0", "#55A868", "#C44E52"][:len(models)])
    plt.ylabel("Mean Inference Time (seconds/sample)")
    plt.title("Inference Time Comparison Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "inference_time.png"), dpi=150)
    plt.close()

    # 3. Memory usage comparison
    plt.figure(figsize=(7, 5))
    plt.bar(models, agg["mean_memory_mb"], color=["#4C72B0", "#55A868", "#C44E52"][:len(models)])
    plt.ylabel("Approx. Peak Memory Usage (MB)")
    plt.title("Memory Usage Comparison Across Models (Approximate)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "memory_usage.png"), dpi=150)
    plt.close()

    # 4. RTF comparison
    plt.figure(figsize=(7, 5))
    plt.bar(models, agg["mean_rtf"], color=["#4C72B0", "#55A868", "#C44E52"][:len(models)])
    plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Real-time threshold (RTF=1.0)")
    plt.ylabel("Mean Real-Time Factor (lower is better)")
    plt.title("Real-Time Factor (RTF) Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "rtf_comparison.png"), dpi=150)
    plt.close()

    # 5. Overall multi-panel comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics_to_plot = [
        ("mean_wer", "Mean WER", axes[0, 0]),
        ("mean_inference_time", "Mean Inference Time (s)", axes[0, 1]),
        ("mean_memory_mb", "Approx. Memory (MB)", axes[1, 0]),
        ("mean_rtf", "Mean RTF", axes[1, 1]),
    ]
    for col, title, ax in metrics_to_plot:
        ax.bar(models, agg[col], color=["#4C72B0", "#55A868", "#C44E52"][:len(models)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Overall Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "comparison.png"), dpi=150)
    plt.close()

    print(f"[visualization] Charts written to {config.RESULTS_DIR}/:")
    for fname in ["wer_comparison.png", "inference_time.png",
                  "memory_usage.png", "rtf_comparison.png", "comparison.png"]:
        print(f"    - {fname}")


if __name__ == "__main__":
    generate_all_charts()
