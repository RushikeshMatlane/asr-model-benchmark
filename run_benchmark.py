"""
run_benchmark.py
=================
Single entry point for the whole pipeline:
  1. Download/load the LibriSpeech subset (if not already cached)
  2. Run all 3 ASR models over every sample, measuring WER/time/memory
  3. Write results/benchmark_results.csv and .json
  4. Generate all comparison charts into results/

Usage:
    python run_benchmark.py
    python run_benchmark.py --num-samples 20
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.benchmark import run_benchmark
from src.visualization import generate_all_charts
from src.report_generator import generate_results_markdown_table


def main():
    parser = argparse.ArgumentParser(description="Run the ASR benchmark pipeline.")
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of audio samples to benchmark (overrides src/config.py NUM_SAMPLES)."
    )
    args = parser.parse_args()

    results = run_benchmark(num_samples=args.num_samples)

    if not results:
        print("\nBenchmark produced no results — see errors above. "
              "Skipping chart generation.")
        sys.exit(1)

    print("\nGenerating charts...")
    generate_all_charts()

    print("\nResults table (paste into report/technical_report.md):\n")
    print(generate_results_markdown_table())

    print("\nDone. See the results/ folder for CSV, JSON, and PNG outputs.")


if __name__ == "__main__":
    main()
