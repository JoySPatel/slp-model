# Room to Speak — Speech Practice Prototype

## Portfolio copy

Built a local Python speech-practice app inspired by my experience with stuttering. Integrated a pretrained WavLM + LoRA model to flag possible speech patterns, added recording and playback, and implemented user-labeled personal training with separate evaluation recordings. Tested classifier-head fine-tuning on public speech data and documented both gains and limitations.

## What I can explain in an interview

- **Motivation:** I wanted a comfortable place to record, listen back, and explore feedback at my own pace.
- **My project:** A Python server, a browser recording/review interface, local model inference, manual clip labeling, and a classifier-head training/evaluation workflow. I used AI assistance during development and should describe that accurately.
- **Model:** An existing WavLM + LoRA checkpoint provides audio representations. I integrated it and experimented with training its classifier head; I did not build or pretrain WavLM.
- **Experiment:** Used 400 public SEP-28k-E training clips and 150 development clips. Development macro F1 increased from 66.8% to 69.4%, while fluent-class F1 decreased from 64.5% to 61.5%.
- **Interpretation:** This is a small development experiment, not an independent test result. The candidate was selected on development performance, and the original checkpoint may already have encountered this dataset.
- **Personalization:** Users can label their own recordings, train on some takes, and compare against separate evaluation takes. Training does not guarantee improvement; the original model remains the default.
- **Next step:** Evaluate on genuinely unseen recordings and investigate missed events and false positives before adopting a candidate.

## Demo sequence

1. Open the app and record a short, freely spoken passage.
2. Play it back and run the experimental pattern suggestions.
3. Show how a suggestion can be corrected or dismissed.
4. Show the public model-development log and explain the development/test distinction.
5. Explain the personal-training workflow without showing private recordings or labels.

Use a fresh browser profile for a portfolio screenshot or video so saved personal sessions are not visible. A demo recording should be one you explicitly intend to share.

## Boundaries

This prototype classifies three-second audio windows. It does not precisely locate every event, distinguish all repetition types, diagnose stuttering, or establish clinical benefit. Do not describe development scores as overall model accuracy or claim personal training necessarily improved detection.

## Included and excluded

This package contains source, tests, model configuration, attribution, and the public-data experiment's aggregate report and training source. It excludes recordings, personal labels, personal reports and weights, public dataset audio, downloaded checkpoints, and screenshots. Download dependencies and the pretrained checkpoint using the README instructions.
