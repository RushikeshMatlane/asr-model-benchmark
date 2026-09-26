"""
dataset_loader.py
==================
Downloads a small subset of LibriSpeech via the HuggingFace `datasets`
library, writes each sample to a local .wav file, and builds a manifest
(CSV) mapping audio_path -> reference_text -> duration -> speaker_id.

This is intentionally "automatic": running this file (or importing
`load_samples()` from it) fetches only NUM_SAMPLES examples, not the full
multi-GB LibriSpeech corpus, and caches results locally so re-runs are fast.
"""

import os
import csv
import sys
import traceback
from dataclasses import dataclass, asdict
from typing import List

import soundfile as sf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


@dataclass
class AudioSample:
    audio_path: str
    reference_text: str
    duration_sec: float
    speaker_id: str
    sample_id: str


def _ensure_dirs():
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)


def download_and_prepare(num_samples: int = None) -> List[AudioSample]:
    """
    Downloads `num_samples` examples from the LibriSpeech validation-clean
    split via HuggingFace `datasets`, saves each as a .wav file under
    data/audio/, and writes a manifest CSV under data/metadata/.

    Returns a list of AudioSample objects.
    """
    num_samples = num_samples or config.NUM_SAMPLES
    _ensure_dirs()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install it with:\n"
            "    pip install datasets\n"
        ) from e

    print(f"[dataset_loader] Loading '{config.DATASET_NAME}' "
          f"({config.DATASET_CONFIG}/{config.DATASET_SPLIT}) — "
          f"streaming first {num_samples} samples...")

    try:
        # Streaming mode avoids downloading the entire split.
        ds = load_dataset(
            config.DATASET_NAME,
            config.DATASET_CONFIG,
            split=config.DATASET_SPLIT,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception:
        print("[dataset_loader] Streaming load failed, retrying with a "
              "non-streaming small split slice...")
        traceback.print_exc()
        ds = load_dataset(
            config.DATASET_NAME,
            config.DATASET_CONFIG,
            split=f"{config.DATASET_SPLIT}[:{num_samples}]",
            trust_remote_code=True,
        )

    samples: List[AudioSample] = []

    for i, example in enumerate(ds):
        if i >= num_samples:
            break
        try:
            audio_array = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            text = example["text"].strip()
            speaker_id = str(example.get("speaker_id", f"unknown_{i}"))
            sample_id = f"sample_{i:04d}"

            out_path = os.path.join(config.AUDIO_DIR, f"{sample_id}.wav")
            sf.write(out_path, audio_array, sr)

            duration_sec = len(audio_array) / sr

            samples.append(AudioSample(
                audio_path=out_path,
                reference_text=text,
                duration_sec=round(duration_sec, 3),
                speaker_id=speaker_id,
                sample_id=sample_id,
            ))
            print(f"  [{i+1}/{num_samples}] saved {sample_id}.wav "
                  f"({duration_sec:.2f}s)")
        except Exception as e:
            print(f"  [WARN] Skipping sample {i} due to error: {e}")
            continue

    if not samples:
        raise RuntimeError(
            "No samples were downloaded. Check your internet connection "
            "and that the 'datasets' package can reach HuggingFace Hub."
        )

    _write_manifest(samples)
    print(f"[dataset_loader] Done. {len(samples)} samples ready in "
          f"{config.AUDIO_DIR}")
    return samples


def _write_manifest(samples: List[AudioSample]):
    with open(config.METADATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))


def load_samples(num_samples: int = None) -> List[AudioSample]:
    """
    Returns samples from the local manifest if it already exists and has
    enough rows; otherwise downloads fresh samples via download_and_prepare().
    """
    num_samples = num_samples or config.NUM_SAMPLES

    if os.path.exists(config.METADATA_FILE):
        samples = []
        with open(config.METADATA_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if os.path.exists(row["audio_path"]):
                    samples.append(AudioSample(
                        audio_path=row["audio_path"],
                        reference_text=row["reference_text"],
                        duration_sec=float(row["duration_sec"]),
                        speaker_id=row["speaker_id"],
                        sample_id=row["sample_id"],
                    ))
        if len(samples) >= num_samples:
            print(f"[dataset_loader] Using {num_samples} cached samples "
                  f"from {config.METADATA_FILE}")
            return samples[:num_samples]

    return download_and_prepare(num_samples)


if __name__ == "__main__":
    samples = load_samples()
    print(f"\nLoaded {len(samples)} samples. First sample:")
    print(samples[0])
