"""
report_generator.py
====================
Optional helper: reads results/aggregate_summary.json (produced by
benchmark.py) and prints a Markdown table you can paste directly into
report/technical_report.md, so you never have to hand-copy numbers.

This does NOT invent numbers — if the aggregate summary doesn't exist yet,
it tells you to run the benchmark first.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def generate_results_markdown_table() -> str:
    agg_path = os.path.join(config.RESULTS_DIR, "aggregate_summary.json")
    if not os.path.exists(agg_path):
        return (
            "**[Benchmark not yet executed — run `python run_benchmark.py` "
            "first, then re-run this script to auto-fill this table.]**"
        )

    with open(agg_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    lines = [
        "| Model | Samples | Mean WER | Median WER | Mean Inference Time (s) | Mean RTF | Approx. Memory (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name, s in summary.items():
        lines.append(
            f"| {model_name} | {s['num_samples']} | {s['mean_wer']} | "
            f"{s['median_wer']} | {s['mean_inference_time']} | "
            f"{s['mean_rtf']} | {s['approx_memory_mb']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_results_markdown_table()
    print(table)

    out_path = os.path.join(config.RESULTS_DIR, "results_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\n[report_generator] Also saved to {out_path}")
