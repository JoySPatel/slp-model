# Training experiments

This folder records actual local training work. The app displays `web/training.json`, written by the training process. It does not simulate progress, results, or improvement. The original model remains active; the candidate checkpoint is a separate research artifact.

## Pilot 001

Fine-tuned the existing model's classifier head for 25 epochs using 400 audio clips from one training shard of SEP-28k-E. WavLM, LoRA, and attentive pooling remained frozen; their real audio features were extracted once. The best head was selected at epoch 4 using macro F1 on 150 development clips. AdamW, learning rate 0.0001, batch size 32, weight decay 0.01, seed 42.

| Development metric | Before | Selected candidate |
|---|---:|---:|
| Macro F1 | 66.8% | 69.4% |
| Accuracy | 66.7% | 69.3% |
| Block F1 | 54.0% | 60.0% |
| Fluent F1 | 64.5% | 61.5% |

The candidate improved some development metrics while increasing some false positives. These results do not demonstrate a clinically useful improvement or unseen-speaker performance. Development-set selection is optimistic; no independent test was used. In particular, the base checkpoint was already trained on SEP-28k, and prior exposure to any selected clip cannot be ruled out.

## Data

Source: [SEP-28k-E authors](https://github.com/th-nuernberg/ml-stuttering-events-dataset-extended), with audio packaged in the [psusac/sep28k mirror](https://huggingface.co/datasets/psusac/sep28k). The original dataset is described by [Lea et al., 2021](https://arxiv.org/abs/2102.12394); the extension and partitioning are described by [Bayerl et al., 2022](https://arxiv.org/abs/2206.03400).

Upstream SEP-28k-E is CC BY-NC 4.0. This is a local noncommercial research experiment. Retain attribution; reassess permissions before commercial use. The mirror does not provide a detailed dataset card. Its provenance has not been independently audited. We preserve its revision and file hashes in `dataset-lock.json`, and retain the upstream README in `DATASET_SOURCE.md`. Audio is not published by this app or committed to source control.

Downloaded train shard: 4,216 rows. Downloaded development split: 4,888 rows. A seeded shuffle selected up to 80 training and 30 development clips per class. Clips with majority uncertainty, poor quality, difficult-to-understand, music, or no-speech labels were excluded. Clips with conflicting fluent/event votes or multiple majority event classes were excluded. Sound and word repetitions were combined. These filters and balancing make this subset easier and less representative than natural recordings.

There are zero shared speaker IDs and identical encoded-audio hashes between the selected local splits. These checks do not rule out mislabeled speakers, re-encoded duplicate audio, or exposure during original checkpoint training. No user's practice recordings were used.

## Reproduce or run another experiment

From the project folder:

```sh
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python setup_training.py
.venv/bin/python train_model.py --run-id pilot-002
```

Use a new run ID; existing runs are protected against overwrite. The current pilot uses CPU feature extraction and takes several minutes on this Mac. It reads only the public parquet files under `training/data/`. It writes progress and the latest completed run to the app log; it does not switch the active inference model. Restart the server after server-code changes, but training-log changes appear during polling or on page refresh.

## Artifacts

`runs/pilot-001/` contains:

- `report.json`: dataset counts, settings, all epochs, before/after metrics, confusion matrices, timestamps, source hashes, and limitations.
- `train_model_snapshot.py`: exact training source used for this run; its SHA-256 is in the report.
- `manifest.json`: selected file IDs, speaker IDs, labels, and audio hashes (local, git-ignored).
- `features.pt`: frozen audio representations and targets (local, git-ignored).
- `candidate-head.pt`: trained classifier weights, base-checkpoint hash, label order, and selected epoch (local, git-ignored).
- `development-predictions.json`: candidate scores for every development clip (local, git-ignored).

Next useful step: establish an independent, speaker-disjoint evaluation dataset with verified absence from the starting model's training data, then examine false positives and missed blocks before deciding whether to deploy the candidate.
