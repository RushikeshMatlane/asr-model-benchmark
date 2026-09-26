# ASR Model Benchmarking Toolkit

A reproducible benchmark comparing three open-source Automatic Speech Recognition (ASR) models — **OpenAI Whisper**, **Faster-Whisper**, and **Wav2Vec2** — for use in a production, noisy-environment voice assistant (e.g. customer-support calls with multiple accents).

## Project Overview

This toolkit downloads a small LibriSpeech audio subset, runs it through all three models under identical conditions, and measures Word Error Rate (WER), inference time, approximate memory usage, and Real-Time Factor (RTF). All comparison charts and the technical report are generated from these measured results — **no benchmark numbers in this repository are fabricated**; anything not yet measured is explicitly labeled `Not yet measured`.

## Features

- Automatic dataset download (no manual file downloads required)
- Common `transcribe(audio_path) -> str` interface across all 3 models
- WER, inference time, RTF, and approximate memory measurement
- CSV + JSON result export
- Automated chart generation (5 comparison charts)
- CPU and NVIDIA GPU support with automatic fallback
- Configurable sample size via a single constant

## Models

| Model | Type | Library |
|---|---|---|
| OpenAI Whisper | Encoder-decoder Transformer (attention-based, autoregressive) | `openai-whisper` |
| Faster-Whisper | Same Whisper architecture, re-implemented on CTranslate2 for faster inference | `faster-whisper` |
| Wav2Vec2 | Self-supervised, CTC-based (encoder-only, non-autoregressive) | `transformers` (`facebook/wav2vec2-base-960h`) |

## Dataset

- **Source**: [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) via HuggingFace `datasets` (`clean` config, `validation` split)
- **Format**: 16 kHz mono WAV, with ground-truth transcripts
- **Size**: configurable, defaults to 50 samples (`NUM_SAMPLES` in `src/config.py`)
- **Note**: LibriSpeech is clean, read audiobook speech — a useful reproducible baseline, but it does **not** represent noisy telephone audio, multiple accents, or overlapping speech typical of customer-support calls. See the technical report's "Noisy Audio Analysis" section for details and an optional noise-augmentation experiment.

## Requirements

- Python 3.9–3.11
- ~5 GB free disk space (model weights + dataset cache)
- FFmpeg on PATH (required by `openai-whisper`; not required by `faster-whisper`)
- Optional: NVIDIA GPU + CUDA for faster inference (CPU works fine for this benchmark scale)

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd asr-benchmark

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install PyTorch (see pytorch.org for your exact CUDA version if using GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
# or for NVIDIA GPU (CUDA 12.1 example):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Windows only, if not already installed) Install FFmpeg
#    via https://ffmpeg.org/download.html and add it to PATH,
#    or: winget install ffmpeg
```

## Usage

```bash
python run_benchmark.py
```

Optional: override the number of samples for a quick smoke test:

```bash
python run_benchmark.py --num-samples 10
```

### Running the Benchmark — what happens

1. Downloads/loads `NUM_SAMPLES` LibriSpeech audio samples into `data/audio/` (cached after first run)
2. Loads each model, transcribes every sample, records timing + memory
3. Writes `results/benchmark_results.csv` and `results/benchmark_results.json`
4. Generates 5 PNG charts into `results/`
5. Prints a Markdown results table you can paste into the technical report

### Expected Output

```
======================================================================
MODEL: whisper  (device=cpu)
======================================================================
[whisper] Loading 'base' model on cpu...
[whisper] Model loaded.
  [1/50] sample_0000: WER=0.043  time=2.31s  RTF=0.41
  ...
[benchmark] whisper approx. memory — initial: 512.3 MB, peak: 891.7 MB, increase: 379.4 MB
...
Results table (paste into report/technical_report.md):
| Model | Samples | Mean WER | ... |
```

## Output Files

| File | Description |
|---|---|
| `results/benchmark_results.csv` | Per-sample, per-model raw results |
| `results/benchmark_results.json` | Same data in JSON form |
| `results/aggregate_summary.json` | Mean/median WER, timing, memory per model |
| `results/results_table.md` | Markdown table for the report |
| `results/wer_comparison.png` | WER bar chart |
| `results/inference_time.png` | Inference time bar chart |
| `results/memory_usage.png` | Memory usage bar chart |
| `results/rtf_comparison.png` | Real-Time Factor bar chart |
| `results/comparison.png` | 4-panel overall comparison |

## Evaluation Metrics

- **WER** = (Substitutions + Deletions + Insertions) / Reference word count, via `jiwer`
- **Inference time**: wall-clock seconds per sample
- **Memory usage**: approximate peak RAM (CPU, via `psutil`) or VRAM (GPU, via `torch.cuda`)
- **RTF (Real-Time Factor)** = inference_time / audio_duration — RTF < 1.0 means faster than real-time

## Results

Populated automatically after running `python run_benchmark.py`. See `results/results_table.md` and `report/technical_report.md` for the full write-up. Until the benchmark has been executed on your machine, all figures are marked `Not yet measured`.

## Project Structure

```
asr-benchmark/
├── data/               # Downloaded audio + metadata manifest
├── models/             # Cached model weights
├── src/                # All source modules (config, models, metrics, benchmark, viz)
├── results/            # CSV/JSON results + generated charts
├── notebooks/          # Optional analysis notebook
├── report/             # Technical report + executive summary
├── requirements.txt
├── run_benchmark.py    # Entry point
└── README.md
```

## Architecture

See `report/technical_report.md` → "Production Architecture" for the full proposed pipeline (audio capture → noise reduction → ASR service → post-processing → LLM/NLP → response), including deployment, scaling, and PII-handling considerations.

## Limitations

- LibriSpeech is clean read-speech; results here are a reproducible baseline, not a guarantee of production performance on noisy telephone audio.
- "Base"/"small" model variants are used for practicality on consumer hardware — larger variants generally improve WER at the cost of latency and memory.
- Memory measurements are approximate (see `src/memory_monitor.py` docstring for caveats).

## Future Improvements

- Add a Common Voice-based noisy/multilingual evaluation pass
- Add the optional noise-augmentation experiment (clean vs. noisy WER)
- Add NeMo ASR / Distil-Whisper as additional candidates
- Add streaming/chunked inference benchmarking for live-call latency

## References

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," arXiv:2212.04356, 2022 — [openai/whisper](https://github.com/openai/whisper)
- SYSTRAN, "faster-whisper: Faster Whisper transcription with CTranslate2" — [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," arXiv:2006.11477, 2020 — [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- Panayotov et al., "Librispeech: An ASR corpus based on public domain audio books," 2015
