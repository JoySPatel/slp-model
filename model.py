"""Local research-model inference. See MODEL_CARD.md for provenance and limits."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent
LABELS = ['Prolongation', 'Repetition', 'Block', 'Interjection', 'Fluent']

class Pooling(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(size, size // 4), nn.Tanh(), nn.Linear(size // 4, 1))
    def forward(self, x):
        weights = self.attention(x).softmax(dim=1)
        mean = (weights * x).sum(dim=1)
        variance = (weights * (x - mean.unsqueeze(1)).square()).sum(dim=1)
        return torch.cat([mean, variance.clamp(min=1e-8).sqrt()], dim=1)

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        config = WavLMConfig.from_dict(json.loads((ROOT / 'models/wavlm-config.json').read_text()))
        self.wavlm = get_peft_model(WavLMModel(config), LoraConfig(r=8, lora_alpha=16,
            lora_dropout=0.1, target_modules=['q_proj', 'v_proj'], bias='none'))
        self.pooling = Pooling(config.hidden_size)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size * 2, 256), nn.GELU(),
            nn.LayerNorm(256), nn.Dropout(0.2), nn.Linear(256, 5))
    def forward(self, audio):
        return self.classifier(self.pooling(self.wavlm(audio).last_hidden_state))

class LocalModel:
    def __init__(self):
        torch.set_num_threads(4)
        self.model = Detector()
        checkpoint = torch.load(ROOT / 'models/stuttering.pt', map_location='cpu', weights_only=True)
        # Strict loading prevents a random or mismatched classification head being used.
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()
    def predict(self, audio, head=None):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not 1600 <= len(audio) <= 48000 or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.01:
            raise ValueError('Expected 0.1–3 seconds of finite mono PCM at 16 kHz in [-1, 1].')
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.003:
            return {'label': 'Quiet audio', 'scores': {}, 'note': 'Low audio level. This does not identify a block.'}
        # Match the training notebook: raw PCM, padded to 3 seconds, no extra normalization.
        padded = np.pad(audio, (0, 48000 - len(audio)))
        with torch.inference_mode():
            tensor = torch.from_numpy(padded).unsqueeze(0)
            logits = self.model(tensor) if head is None else head(self.model.pooling(self.model.wavlm(tensor).last_hidden_state))
            scores = logits.softmax(-1)[0].tolist()
        return {'label': LABELS[int(np.argmax(scores))], 'scores': dict(zip(LABELS, scores)),
                'note': 'Experimental window-level suggestion; scores are not calibrated certainty.'}
