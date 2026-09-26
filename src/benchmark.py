"""
benchmark.py
============
Orchestrates the full benchmark: runs every model over every audio sample,
measuring WER, inference time, and approximate memory usage, then writes
results to CSV and JSON, plus prints aggregate statistics.

Run directly:
    python -m src.benchmark
or via the project root entry point:
    python run_benchmark.py
"""

import os
import sys
import time
import json
import csv
import statistics as stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from dataset_loader import load_samples
from metrics import calculate_wer, calculate_rtf
from memory_monitor import MemoryMonitor

from whisper_model import WhisperASR
from faster_whisper_model import FasterWhisperASR
from wav2vec2_model import Wav2Vec2ASR


def get_models():
    """Returns the list of model wrapper instances to benchmark."""
    return [
        WhisperASR(),
        FasterWhisperASR(),
        Wav2Vec2ASR(),
    ]


def run_benchmark(num_samples: int = None):
    num_samples = num_samples or config.NUM_SAMPLES
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ASR BENCHMARK — loading dataset")
    print("=" * 70)
    samples = load_samples(num_samples)

    results = []

    for model in get_models():
        print("\n" + "=" * 70)
        print(f"MODEL: {model.name}  (device={config.DEVICE})")
        print("=" * 70)

        try:
            model.load()
        except Exception as e:
            print(f"[benchmark] FAILED to load {model.name}: {e}")
            print(f"[benchmark] Skipping {model.name} for all samples.")
            continue

        mon = MemoryMonitor(device=config.DEVICE)
        mon.start()

        for i, sample in enumerate(samples):
            try:
                start = time.perf_counter()
                predicted_text = model.transcribe(sample.audio_path)
                inference_time = time.perf_counter() - start

                mon.sample()
                wer = calculate_wer(sample.reference_text, predicted_text)
                rtf = calculate_rtf(inference_time, sample.duration_sec)

                row = {
                    "model": model.name,
                    "audio_file": os.path.basename(sample.audio_path),
                    "audio_duration": sample.duration_sec,
                    "reference_text": sample.reference_text,
                    "predicted_text": predicted_text,
                    "wer": round(wer, 4),
                    "inference_time": round(inference_time, 4),
                    "rtf": round(rtf, 4),
                    "memory_usage_mb": None,  # filled in after the loop (peak for this model)
                    "device": config.DEVICE,
                }
                results.append(row)

                print(f"  [{i+1}/{len(samples)}] {sample.sample_id}: "
                      f"WER={wer:.3f}  time={inference_time:.2f}s  "
                      f"RTF={rtf:.2f}")
            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] ERROR on "
                      f"{sample.sample_id}: {e}")
                continue

        mem_stats = mon.stop()
        # Backfill this model's memory usage for all its rows in this run
        for row in results:
            if row["model"] == model.name and row["memory_usage_mb"] is None:
                row["memory_usage_mb"] = mem_stats["peak_mb"]

        print(f"[benchmark] {model.name} approx. memory — "
              f"initial: {mem_stats['initial_mb']} MB, "
              f"peak: {mem_stats['peak_mb']} MB, "
              f"increase: {mem_stats['increase_mb']} MB")

        model.unload()

    if not results:
        print("\n[benchmark] No results were produced. Check the error "
              "messages above (likely a missing dependency or failed "
              "model download).")
        return []

    _write_csv(results)
    _write_json(results)
    _print_aggregates(results)
    return results


def _write_csv(results):
    fieldnames = list(results[0].keys())
    with open(config.RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[benchmark] Wrote {config.RESULTS_CSV}")


def _write_json(results):
    with open(config.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] Wrote {config.RESULTS_JSON}")


def _print_aggregates(results):
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (per model)")
    print("=" * 70)

    by_model = {}
    for row in results:
        by_model.setdefault(row["model"], []).append(row)

    aggregate_summary = {}
    for model_name, rows in by_model.items():
        wers = [r["wer"] for r in rows]
        times = [r["inference_time"] for r in rows]
        rtfs = [r["rtf"] for r in rows]
        mem = rows[0]["memory_usage_mb"]
        total_time = sum(times)

        summary = {
            "num_samples": len(rows),
            "mean_wer": round(stats.mean(wers), 4),
            "median_wer": round(stats.median(wers), 4),
            "mean_inference_time": round(stats.mean(times), 4),
            "median_inference_time": round(stats.median(times), 4),
            "mean_rtf": round(stats.mean(rtfs), 4),
            "approx_memory_mb": mem,
            "total_processing_time_sec": round(total_time, 2),
        }
        aggregate_summary[model_name] = summary

        print(f"\n{model_name}:")
        for k, v in summary.items():
            print(f"    {k}: {v}")

    agg_path = os.path.join(config.RESULTS_DIR, "aggregate_summary.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2)
    print(f"\n[benchmark] Wrote {agg_path}")


if __name__ == "__main__":
    run_benchmark()
