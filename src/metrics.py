"""
metrics.py
==========
Word Error Rate (WER) calculation and text normalization, plus Real-Time
Factor (RTF) helper.

WER = (S + D + I) / N
  S = substitutions, D = deletions, I = insertions
  N = number of words in the reference transcript

Lower WER is better: it means fewer word-level edits are needed to turn the
predicted transcript into the reference transcript, i.e. higher transcription
accuracy. WER of 0.0 = perfect match; WER can exceed 1.0 if the hypothesis
has many extra inserted words relative to a short reference.

We rely on the `jiwer` library for the underlying edit-distance alignment
rather than reimplementing it, since jiwer is a well-tested standard tool
for ASR evaluation.
"""

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def normalize_text(text: str) -> str:
    """
    Normalizes a transcript before WER comparison:
    - lowercases
    - removes punctuation (keeps apostrophes inside words, e.g. "it's")
    - collapses whitespace

    Both reference and hypothesis MUST be normalized the same way, or WER
    will be inflated by superficial formatting differences rather than
    actual transcription errors.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Returns the Word Error Rate between a reference and hypothesis string,
    after normalizing both. Requires the `jiwer` package.
    """
    try:
        import jiwer
    except ImportError as e:
        raise ImportError(
            "jiwer is not installed. Run:\n    pip install jiwer\n"
        ) from e

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if len(ref_norm) == 0:
        # Avoid division by zero; an empty reference with a non-empty
        # hypothesis is treated as 100% error, both empty as 0% error.
        return 0.0 if len(hyp_norm) == 0 else 1.0

    return jiwer.wer(ref_norm, hyp_norm)


def calculate_rtf(inference_time_sec: float, audio_duration_sec: float) -> float:
    """
    Real-Time Factor (RTF) = inference_time / audio_duration

    RTF < 1.0 means the model transcribes FASTER than real-time (e.g. RTF
    of 0.3 means a 10s clip takes 3s to transcribe) — required for live
    voice-assistant use. RTF >= 1.0 means the model is too slow for live
    use and can only be used for offline/batch transcription.
    """
    if audio_duration_sec <= 0:
        return float("nan")
    return inference_time_sec / audio_duration_sec


def wer_example():
    """Prints a worked WER example for the technical report / README."""
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the quick brown fox jump over lazy dog"
    wer = calculate_wer(reference, hypothesis)
    print(f"Reference : {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"WER       : {wer:.3f}")
    print("(1 substitution 'jumps'->'jump', 1 deletion 'the' before 'lazy')")
    return wer


if __name__ == "__main__":
    wer_example()
