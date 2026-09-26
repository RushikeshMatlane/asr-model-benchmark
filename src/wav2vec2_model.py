"""
wav2vec2_model.py
==================
Wrapper around HuggingFace Transformers' Wav2Vec2 (facebook/wav2vec2-base-960h
by default) — a CTC-based (encoder-only, no autoregressive decoder) ASR model.
Implements the common ASRModel interface.

Reference: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised
Learning of Speech Representations" (arXiv:2006.11477), Facebook AI, 2020.

Install:
    pip install transformers torch librosa soundfile
"""

import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from transcription import ASRModel


class Wav2Vec2ASR(ASRModel):
    name = "wav2vec2"

    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or config.WAV2VEC2_MODEL_ID
        self.device = device or config.DEVICE
        self.model = None
        self.processor = None

    def load(self):
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as e:
            raise ImportError(
                "transformers/torch are not installed. Run:\n"
                "    pip install transformers torch\n"
            ) from e

        print(f"[wav2vec2] Loading '{self.model_id}' on {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.model_id, cache_dir=config.MODEL_CACHE_DIR
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id, cache_dir=config.MODEL_CACHE_DIR
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Wav2Vec2 model '{self.model_id}'. Check "
                f"your internet connection (first run downloads weights "
                f"from HuggingFace Hub). Original error: {e}"
            ) from e
        print("[wav2vec2] Model loaded.")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            raise RuntimeError("Call .load() before .transcribe().")
        try:
            import torch
            import librosa

            speech, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE)
            inputs = self.processor(
                speech, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
            )
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcript = self.processor.batch_decode(predicted_ids)[0]
            return transcript.strip().lower()
        except Exception as e:
            print(f"[wav2vec2] ERROR transcribing {audio_path}: {e}")
            return ""

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    from dataset_loader import load_samples

    samples = load_samples(num_samples=1)
    model = Wav2Vec2ASR()
    model.load()
    text = model.transcribe(samples[0].audio_path)
    print("Reference:", samples[0].reference_text)
    print("Predicted:", text)
