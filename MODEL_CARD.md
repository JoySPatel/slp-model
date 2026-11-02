# Research model integration

## Provenance

- Checkpoint: [pmootr/stuttering-detection-wavlm-lora](https://huggingface.co/pmootr/stuttering-detection-wavlm-lora), revision `7bf27ef2e63c88ad673a7696ac0469c0f8e66e91`, file `best_model_wavlm_lora.pt`.
- Weights are listed as Apache-2.0 by the publisher. They are downloaded separately, not committed to this project.
- SHA-256: `752de4ef57d7afcbdd40ef6a1770923a9dcedb7557f008134697508b4f151a94`.
- Backbone configuration: [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus), bundled locally in `models/wavlm-config.json`.
- Architecture, label order, and preprocessing checked against the author's [training notebook](https://github.com/keyboardban/stuttering-detection-wavlm-lora/blob/main/02-wavlm-lora.ipynb). This app does not execute that notebook or download remote Python code.

The model card's quickstart and repository inference example do not match the published full checkpoint. This integration reconstructs WavLM with rank-8 LoRA on q/v projections, attentive statistics pooling, and the trained two-layer classification head. Every state-dictionary key must match (`strict=True`). Loading uses `torch.load(weights_only=True)`. The full checkpoint includes backbone weights, so no second backbone download is required.

## Input and output

Browser audio is decoded and resampled to 16 kHz mono using Web Audio. It is sent in three-second, non-overlapping windows to the loopback Python server as float32 PCM. Short final windows are padded to three seconds, matching the training notebook. Tails shorter than 0.1 seconds are not classified. No extra waveform normalization is applied.

Label order: Prolongation, Repetition, Block, Interjection, Fluent. The model combines sound and word repetitions. Each window receives one highest-scoring class, not a precise event boundary or count. Windows with model label Fluent are summarized as unflagged, not proof of absence of stuttering. Other windows appear as review candidates. A simple RMS gate below 0.003 labels very quiet windows as "Quiet audio", never as a confirmed block. Quiet windows remain reviewable because silent blocks can occur there. This gate is a heuristic, not a speech detector; tones, music, breathing, and noise may still produce misleading predictions.

Softmax scores are available under details but are not calibrated probabilities of an actual event. The same event may span multiple windows; multiple event types can occur within one window. No overall fluency, severity, progress, or diagnosis score is computed.

## Evidence and limitations

The publisher reports weighted F1 around 0.67 and block F1 around 0.36 on its evaluation data. These are publisher claims, not this app's measured accuracy. The training notebook uses clip-level splitting; speaker-independent generalization has not been established here. English podcast training data may not represent a user's voice, accent, recording conditions, or stuttering experience. This app has not undergone clinical validation or an accuracy evaluation on annotated user recordings.

[SEP-28k research](https://arxiv.org/abs/2102.12394) provides the event-detection context. [ASHA's clinical guidance](https://www.asha.org/practice-portal/clinical-topics/fluency-disorders/) distinguishes stuttering features and the person's broader experience. A [2025 study](https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.pdf) demonstrates why ordinary speech transcription is not sufficient for preserving disfluencies.

## Privacy and validation

Inference runs on CPU locally, with no network calls during model loading or inference. The server binds only to 127.0.0.1, validates Host and Origin, and accepts only bounded PCM windows. Audio is not logged or written to disk by the server. Python and browser memory may remain subject to ordinary operating-system memory management; this is not a forensic erasure guarantee.

Browser recordings are initially memory-only. Explicit Save stores the recording, annotations, and notes in IndexedDB for that exact browser/origin. Stored data is not encrypted. Downloads are user-chosen exports. There is no telemetry or cloud sync. External source links open only if clicked.

Verification includes strict weight loading, finite input-dependent inference, validation of malformed PCM, and silence handling. Synthetic signals verify plumbing, not clinical performance. Microphone recording requires testing on the user's actual browser/device and its permission settings.

## Local fine-tuning pilot

The project now includes a completed classifier-head fine-tuning experiment on a filtered SEP-28k-E subset. This is an internal development experiment, not an independent accuracy benchmark: the starting model's prior exposure to SEP-28k clips is unknown. The original checkpoint remains active. See [training/README.md](training/README.md) and the app's Model development section for actual results, regressions, provenance, and artifacts.
