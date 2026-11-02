# Room to Speak

A private speech-practice prototype inspired by lived experience with stuttering. Record at your own pace, replay moments, review experimental model suggestions, and save your own reflections. Python runs the audio model locally; the browser provides recording and review.

## Open it on this Mac

Double-click **Start.command** in this folder, or use Terminal:

```sh
cd room-to-speak
.venv/bin/python app.py
```

Open **http://127.0.0.1:8765** in Chrome or Safari. Keep the Terminal process running. Press Ctrl+C to stop it. If the port is already in use, an existing copy may already be running. To use another port: `.venv/bin/python app.py --port 8766` (saved sessions are specific to the browser and port).

Complete the installation steps below on first use. The model takes a few seconds to load. No network connection is required after installation.

## Try it

1. Choose a prompt, or speak freely.
2. Press **Start recording**, allow your browser's microphone request, then **Finish recording** whenever you choose. There is no silence-triggered cutoff.
3. Listen back. Optionally press **Find possible patterns** to run the local research model.
4. Replay windows and correct or dismiss suggestions. Add a moment the model missed, including your own distinction between sound and whole-word repetition.
5. Add a reflection and choose **Save on this device**, **Download recording**, or **Discard this session**.

You can import an audio file supported by your browser (maximum import size 100 MB). Short practice recordings are recommended for manageable memory use; recording is held in browser memory until stopped. Audio is analyzed in three-second windows, with progress and a stop-analysis control.

## What is implemented

- Browser microphone recording, audio import, playback, and waveform.
- Local WavLM + LoRA inference for potential repetition, prolongation, block, and interjection windows.
- A quiet-audio review category that does not equate silence with a block.
- Manual timed observations and corrections, dismiss/restore, and session reflections.
- Explicit save/load/delete using browser IndexedDB; audio download.
- A loopback-only Python server with same-origin checks and bounded input validation.

The pretrained research model is integrated; this project additionally implements classifier-head fine-tuning, described below. It does not distinguish sound repetition from word repetition automatically. It is a window classifier, not an exact event locator. Its block detection is particularly weak. Outputs are experimental suggestions, not a diagnosis, therapy, or a validated measurement of improvement. See [MODEL_CARD.md](MODEL_CARD.md) for provenance, reported performance, and limitations.

## Install on another computer

Python 3.10+ with compatible PyTorch wheels is required (tested on this Mac with Python 3.13).

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python setup_model.py
.venv/bin/python app.py
```

The setup downloads approximately 381 MB of model weights plus dependencies. The checksum is verified. `models/wavlm-config.json` is included. On Windows, use `.venv\Scripts\python` and `.venv\Scripts\pip` instead. `Start.command` is macOS-specific.

## Privacy

No recordings are sent to a third-party service. Analysis sends PCM only to the Python process on the same computer, in memory. Sessions are saved only when requested, in the browser's IndexedDB, and are not encrypted. Anyone with access to that browser profile may access them. Clearing browser data deletes saved sessions. Use Download recording for recordings you want to keep independently. Deleting a saved session does not delete any separate downloads or an already open in-memory copy.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests -v
node --check web/app.js
```

Tests check input handling, model loading and input-dependent outputs, silence handling, and the earlier transcript analyzer. They do not establish detection accuracy on real stuttering. The browser flow is also checked with synthetic audio. Live microphone behavior still depends on browser/device permissions.

## Files

- `app.py`: local server, model status, and in-memory analysis endpoint.
- `model.py`: architecture reconstruction and inference.
- `web/`: recording, review, session storage, and interface.
- `setup_model.py`: pinned model download with checksum verification.
- `MODEL_CARD.md`: source attribution and limitations.
- `speech_feedback.py`: earlier standalone transcript-analysis CLI, retained as a separate utility. It is not used for audio stuttering detection.

## Model development

The app now displays a real training experiment log. A local pilot fine-tuned the classifier head on 400 public SEP-28k-E clips and selected a candidate using 150 development clips. The candidate is saved separately; the original model is still active. No personal recordings are used. See [training/README.md](training/README.md) for reproduction commands and evaluation limits. The base model was integrated from an existing research project; only this additional classifier-head training was performed here.

## Personal training

1. In **Personal training**, upload recordings or explicitly add the currently open practice recording. Each added copy is saved in a separate browser IndexedDB database.
2. Expand each recording, listen to each 3-second clip, and assign a label. Use Skip for uncertain, mixed-event, or non-speech windows. Repetition combines sound and whole-word repetitions.
3. Assign at least two distinct recordings to Training and another to Evaluation. Minimum: 10 training clips, 5 evaluation clips, and at least two training label types. Evaluation label types missing from personal training are allowed and explicitly listed in the results. These thresholds enable a small prototype run; they do not establish statistical reliability.
4. Check the explicit consent box and press **Train personal model**. Maximum 30 recordings, 5 minutes per recording, and 10 minutes total per run. Keep the local server running.
5. Review original-versus-personal evaluation results. Choose **Personal model (experimental)** above Analyze recording if you want to try it. The original model remains the default and can always be selected.

The fixed 15-epoch run trains only the classifier head, with the encoder and pooling frozen. Evaluation recordings do not update weights or select an epoch. Exact duplicate recordings and matching quantized clips across splits are rejected; this cannot detect all re-encoded or overlapping copies. Use genuinely different takes, ideally recorded at different times. Repeated experiments on the same evaluation recordings weaken their independence.

Audio is stored only in the browser queue and sent to the loopback Python process in memory when training is requested. The server persists model heads and a report under git-ignored `personal_models/`, but no raw personal audio. Weights can encode information about training data; treat that folder as private. Browser data and model files are not encrypted. Removing a queue recording does not remove older trained weights, practice copies, or downloads. To remove model artifacts, stop the server and delete the personal_models folder yourself. Clearing browser data removes the labeling queue.

Automated synthetic-audio tests cover consent, labels, whole-recording splits, duplicate rejection, a complete training run, checkpoint loading, and inference. They do not measure actual stuttering-detection quality. No personal model has been trained on the user's recordings automatically.
