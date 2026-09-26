"""
faster_whisper_model.py
========================
Wrapper around SYSTRAN's faster-whisper (CTranslate2 reimplementation of
Whisper). Implements the common ASRModel interface.

faster-whisper claims up to ~4x faster inference than openai/whisper at
equivalent accuracy, with lower memory usage, especially with int8
quantization (SYSTRAN/faster-whisper, MIT license). This benchmark measures
whether that holds on your hardware and audio — do not assume it without
running the benchmark.

Install:
    pip install faster-whisper
    (No system FFmpeg required — bundles PyAV/FFmpeg internally.)
"""

import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from transcription import ASRModel


class FasterWhisperASR(ASRModel):
    name = "faster_whisper"

    def __init__(self, model_size: str = None, device: str = None,
                 compute_type: str = None):
        self.model_size = model_size or config.FASTER_WHISPER_MODEL_SIZE
        self.device = device or config.DEVICE
        self.compute_type = compute_type or config.FASTER_WHISPER_COMPUTE_TYPE
        self.model = None

    def load(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Run:\n"
                "    pip install faster-whisper\n"
            ) from e

        print(f"[faster_whisper] Loading '{self.model_size}' model on "
              f"{self.device} (compute_type={self.compute_type})...")
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=config.MODEL_CACHE_DIR,
            )
        except Exception as e:
            # Common failure: GPU compute_type unsupported by installed
            # CTranslate2/cuDNN version. Fall back to CPU int8.
            print(f"[faster_whisper] Load failed on {self.device} "
                  f"({e}). Falling back to CPU/int8.")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=config.MODEL_CACHE_DIR,
            )
        print("[faster_whisper] Model loaded.")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Call .load() before .transcribe().")
        try:
            segments, _info = self.model.transcribe(audio_path, beam_size=5)
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"[faster_whisper] ERROR transcribing {audio_path}: {e}")
            return ""

    def unload(self):
        del self.model
        self.model = None
        gc.collect()


if __name__ == "__main__":
    from dataset_loader import load_samples

    samples = load_samples(num_samples=1)
    model = FasterWhisperASR()
    model.load()
    text = model.transcribe(samples[0].audio_path)
    print("Reference:", samples[0].reference_text)
    print("Predicted:", text)
