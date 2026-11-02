"""Fetch pinned public research data; never accesses personal session storage."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

if __name__=='__main__':
    config=json.loads((ROOT/'training/dataset-lock.json').read_text())
    folder=ROOT/'training/data';folder.mkdir(parents=True,exist_ok=True)
    for name,info in config['files'].items():
        path=folder/f'{name}.parquet'
        if path.exists() and sha(path)==info['sha256']:
            print(f'{name}: verified');continue
        url=f"https://huggingface.co/datasets/psusac/sep28k/resolve/{config['revision']}/{info['path']}"
        temp=path.with_suffix('.part')
        try:
            print(f'Downloading {name} research split…',flush=True)
            urllib.request.urlretrieve(url,temp)
            if sha(temp)!=info['sha256']:raise RuntimeError('Dataset checksum mismatch')
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)
