"""
transcription.py
=================
Defines the common interface every ASR model wrapper must implement, so
benchmark.py can treat Whisper, Faster-Whisper, and Wav2Vec2 interchangeably.
"""

from abc import ABC, abstractmethod


class ASRModel(ABC):
    """Common interface for all ASR model wrappers used in this benchmark."""

    name: str = "base_model"

    @abstractmethod
    def load(self):
        """Load model weights into memory. Called once before benchmarking."""
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.

        Args:
            audio_path: path to a .wav file (mono, 16kHz recommended).

        Returns:
            The predicted transcript as a plain string.
        """
        raise NotImplementedError

    def unload(self):
        """
        Optional: free model memory (useful between models in a long
        benchmark run so peak memory measurements aren't contaminated by
        the previous model still being resident).
        """
        pass
