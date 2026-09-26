"""
whisper_model.py
=================
Wrapper around OpenAI's original Whisper implementation (pip package
"openai-whisper"). Implements the common ASRModel interface.

Install:
    pip install openai-whisper
    (requires FFmpeg on PATH — see README / setup instructions)
"""

import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from transcription import ASRModel


class WhisperASR(ASRModel):
    name = "whisper"

    def __init__(self, model_size: str = None, device: str = None):
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.device = device or config.DEVICE
        self.model = None

    def load(self):
        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "openai-whisper is not installed. Run:\n"
                "    pip install openai-whisper\n"
            ) from e

        print(f"[whisper] Loading '{self.model_size}' model on "
              f"{self.device}...")
        try:
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=config.MODEL_CACHE_DIR,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_size}'. "
                f"Check your internet connection (first run downloads "
                f"weights) and that FFmpeg is installed. Original error: {e}"
            ) from e
        print("[whisper] Model loaded.")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Call .load() before .transcribe().")
        try:
            # fp16 only makes sense on GPU
            result = self.model.transcribe(
                audio_path,
                fp16=(self.device == "cuda"),
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[whisper] ERROR transcribing {audio_path}: {e}")
            return ""

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    # Quick smoke test
    from dataset_loader import load_samples

    samples = load_samples(num_samples=1)
    model = WhisperASR()
    model.load()
    text = model.transcribe(samples[0].audio_path)
    print("Reference:", samples[0].reference_text)
    print("Predicted:", text)
