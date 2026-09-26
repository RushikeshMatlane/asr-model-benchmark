# Executive Summary: Speech-to-Text Model Evaluation for Customer-Support Voice AI

## Business Problem

Our hypothetical voice-based customer-support assistant needs a speech-to-text (ASR) engine that works reliably on real calls — which are often noisy, come from callers with different accents, and demand near-instant responses. Choosing the wrong engine risks poor transcription accuracy, slow response times, or high infrastructure costs at scale.

## Models Evaluated

We evaluated three widely used, open-source speech-to-text models:

- **OpenAI Whisper** — a general-purpose model trained on a very large, diverse set of internet audio.
- **Faster-Whisper** — the same Whisper model, re-engineered for faster inference.
- **Wav2Vec2** — a leaner model trained differently, well suited to further customization for a specific domain.

## Dataset

Models were tested on a small, standard, publicly available speech dataset (LibriSpeech) so results are reproducible and comparable. This dataset is clean, native-English audio — a good starting baseline, but not a full stand-in for noisy real call audio.

## Main Evaluation Metrics

- **Accuracy** (Word Error Rate — lower is better)
- **Speed** (transcription time, and whether it keeps up with a live conversation)
- **Resource usage** (approximate memory footprint)
- **Ease of deployment**

## Key Findings

**Benchmark status: pending execution.** As of this document, the benchmarking code has been built and is ready to run, but has not yet been executed on production-representative hardware. In line with our commitment to evidence-based decisions, we are not reporting estimated or assumed performance numbers. Once executed, this section will summarize measured accuracy, speed, and resource usage for all three models side by side.

What we can say from model research alone: Whisper's much larger and more varied training data suggests it may generalize better to accents and background noise than Wav2Vec2's current configuration, while Faster-Whisper claims meaningful speed and memory advantages over standard Whisper at the same accuracy. **These are expectations from public documentation, not yet confirmed by our own testing.**

## Deployment Considerations

- All three models can run on standard servers; only Faster-Whisper's GPU path requires careful version matching with NVIDIA libraries.
- None of the three models were tested yet on telephone-quality or noisy audio — a necessary next step before a production decision.
- Data privacy (call audio, PII in transcripts) must be addressed in the production pipeline regardless of which model is chosen.

## Recommendation

**Recommendation pending benchmark execution.** A final model choice will be made only after running the completed benchmark on representative hardware and, ideally, on a noisier evaluation set closer to real call conditions.

## Next Steps

1. Execute the benchmark (`python run_benchmark.py`) on target hardware.
2. Review measured accuracy, speed, and memory results.
3. Run a secondary evaluation on noisier, more accent-diverse audio (e.g., Common Voice) before finalizing the recommendation.
4. Proceed to fine-tuning and production-architecture planning for the selected model.
