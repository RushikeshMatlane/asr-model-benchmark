"""
config.py
=========
Single source of truth for benchmark settings. Change values here to
resize or reconfigure the whole benchmark without touching other files.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------
# Number of audio samples to pull from the dataset for this benchmark run.
# Increase for a more statistically meaningful result; decrease for a fast
# smoke-test while developing.
NUM_SAMPLES = 50

# HuggingFace dataset identifier and split/config used for the LibriSpeech
# subset. "clean" is the easier, low-noise subset; "other" is the harder,
# noisier subset. We default to "clean" for reproducibility, and note in the
# report that this under-represents real customer-support noise conditions.
DATASET_NAME = "librispeech_asr"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"  # small split, good for quick benchmarking

# Where downloaded/extracted audio and metadata are cached locally.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
METADATA_FILE = os.path.join(METADATA_DIR, "manifest.csv")

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# Practical small/base variants so this runs on a normal laptop.
WHISPER_MODEL_SIZE = "base"                     # tiny, base, small, medium, large
FASTER_WHISPER_MODEL_SIZE = "base"              # same size scale as openai/whisper
WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base-960h"

MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# ---------------------------------------------------------------------------
# Device settings
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Returns 'cuda' if an NVIDIA GPU is available and usable, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

# Faster-Whisper uses CTranslate2 compute types rather than raw torch dtypes.
FASTER_WHISPER_COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "float16"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SAMPLE_RATE = 16000  # all models in this project expect 16kHz mono audio
