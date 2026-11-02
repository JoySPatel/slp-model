"""Download pinned public weights only. Never uploads audio."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/pmootr/stuttering-detection-wavlm-lora/resolve/7bf27ef2e63c88ad673a7696ac0469c0f8e66e91/best_model_wavlm_lora.pt'
SHA256 = '752de4ef57d7afcbdd40ef6a1770923a9dcedb7557f008134697508b4f151a94'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    target = ROOT / 'models/stuttering.pt'
    target.parent.mkdir(exist_ok=True)
    if target.exists() and digest(target) == SHA256:
        print('Model already installed and verified.')
    else:
        temporary = target.with_suffix('.part')
        print('Downloading research model (~381 MB). No audio is uploaded.', flush=True)
        try:
            urllib.request.urlretrieve(URL, temporary)
            if digest(temporary) != SHA256: raise RuntimeError('Model checksum mismatch; download rejected.')
            temporary.replace(target)
            print('Model installed and verified.')
        finally:
            temporary.unlink(missing_ok=True)
