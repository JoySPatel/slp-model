"""Reproducible classifier-head fine-tuning pilot; never reads personal recordings."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
import torch
from model import LocalModel, LABELS

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'training/runs/pilot-001'
REPORT = ROOT / 'web/training.json'
SEED = 42

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_report(report):
    report['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = REPORT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(REPORT)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN/'report.json').write_text(REPORT.read_text())
    print(report['stage'], flush=True)

def target(row):
    if any(row[key] >= 2 for key in ['Unsure','PoorAudioQuality','DifficultToUnderstand','Music','NoSpeech']): return None
    scores=[row['Prolongation'],max(row['SoundRep'],row['WordRep']),row['Block'],row['Interjection']]
    positives=[i for i,v in enumerate(scores) if v>=2]
    # Reject conflicting/multiple event labels, rather than inventing a priority order.
    if len(positives)==1 and row['NoStutteredWords']<2:return positives[0]
    if not positives and row['NoStutteredWords']>=2:return 4
    return None

def select(path, per_class, excluded_speakers=set(), excluded_hashes=set()):
    rows=pq.read_table(path).to_pylist();random.Random(SEED).shuffle(rows)
    selected=[];counts=[0]*5;seen=set();skipped=0
    for row in rows:
        label=target(row);speaker=row['speaker']
        if label is None or not speaker or speaker in excluded_speakers or counts[label]>=per_class:
            skipped+=1;continue
        digest=hashlib.sha256(row['audio']['bytes']).hexdigest()
        if digest in seen or digest in excluded_hashes:skipped+=1;continue
        seen.add(digest);counts[label]+=1
        selected.append({'row':row,'label':label,'hash':digest})
    if min(counts)<10:raise RuntimeError(f'Insufficient examples per class: {counts}')
    return selected,{'available_rows':len(rows),'selected':len(selected),'class_counts':dict(zip(LABELS,counts)),'skipped_or_not_selected':skipped}

def metrics(labels, logits):
    prediction=np.asarray(logits).argmax(axis=1);labels=np.asarray(labels)
    matrix=np.zeros((5,5),dtype=int)
    for truth,pred in zip(labels,prediction):matrix[truth,pred]+=1
    per={}
    for i,name in enumerate(LABELS):
        tp=int(matrix[i,i]);fp=int(matrix[:,i].sum()-tp);fn=int(matrix[i,:].sum()-tp)
        per[name]={'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'support':int(matrix[i].sum())}
    return {'accuracy':float((labels==prediction).mean()),'macro_f1':float(np.mean([v['f1'] for v in per.values()])), 'per_class':per,'confusion_matrix':matrix.tolist()}

def extract(model, examples, split, report):
    features=[];labels=[]
    for index,item in enumerate(examples):
        audio,sr=sf.read(io.BytesIO(item['row']['audio']['bytes']),dtype='float32',always_2d=True)
        audio=audio.mean(axis=1)
        if sr!=16000:
            divisor=math.gcd(sr,16000);audio=resample_poly(audio,16000//divisor,sr//divisor).astype(np.float32)
        audio=audio[:48000];audio=np.pad(audio,(0,max(0,48000-len(audio))))
        if not np.isfinite(audio).all():raise ValueError('Non-finite dataset audio')
        with torch.no_grad():
            hidden=model.wavlm(torch.from_numpy(audio).unsqueeze(0)).last_hidden_state
            features.append(model.pooling(hidden).squeeze(0).detach())
        labels.append(item['label'])
        if (index+1)%25==0 or index+1==len(examples):
            report['stage']=f'Extracting {split} audio features: {index+1}/{len(examples)}'
            report['processed_clips']=report.get('processed_clips',0)+(25 if (index+1)%25==0 else (index+1)%25)
            write_report(report)
    return torch.stack(features),torch.tensor(labels,dtype=torch.long)

def main():
    if RUN.exists():raise RuntimeError('pilot-001 already exists. Preserve the run; choose a new RUN and run_id for a new experiment.')
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    report={'run_id':'pilot-001','status':'running','stage':'Preparing dataset','started_at':datetime.now(timezone.utc).isoformat(),
        'dataset':'SEP-28k-E · psusac/sep28k mirror','dataset_url':'https://huggingface.co/datasets/psusac/sep28k',
        'dataset_revision':'67630930030046543194dc7c701b87b2cdc7680f','license':'CC BY-NC 4.0 (upstream SEP-28k-E)',
        'method':'Fine-tune classifier head; freeze WavLM, LoRA, and attentive pooling.','active_model':'Original research checkpoint (unchanged)',
        'limitations':['The starting model was already trained on SEP-28k. Prior exposure to these clips is unknown; this is not an unseen-data benchmark.',
        'A small, class-balanced subset with ambiguous/multi-event clips excluded. Results do not represent natural event frequencies.',
        'Development data selects the best epoch. No independent final test or clinical validation has been performed.',
        'Speaker IDs are dataset metadata, not independently verified. No personal recordings are used.'],
        'config':{'seed':SEED,'epochs':25,'learning_rate':0.0001,'batch_size':32,'optimizer':'AdamW','weight_decay':0.01,'training_cap_per_class':80,'development_cap_per_class':30},'history':[]}
    write_report(report)
    try:
        train,train_info=select(ROOT/'training/data/train.parquet',80)
        train_speakers={i['row']['speaker'] for i in train};train_hashes={i['hash'] for i in train}
        dev,dev_info=select(ROOT/'training/data/dev.parquet',30,train_speakers,train_hashes)
        assert not train_speakers & {i['row']['speaker'] for i in dev}
        assert not train_hashes & {i['hash'] for i in dev}
        report.update(train=train_info,development=dev_info,speaker_overlap=0,audio_hash_overlap=0)
        manifest=[]
        for split,examples in [('train',train),('development',dev)]:
            for item in examples:manifest.append({'split':split,'file':item['row']['audio']['path'],'speaker':item['row']['speaker'],'audio_sha256':item['hash'],'label':LABELS[item['label']]})
        (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
        report['data_sha256']={name:sha(ROOT/f'training/data/{name}.parquet') for name in ('train','dev')}
        report['base_model_sha256']=sha(ROOT/'models/stuttering.pt')
        report['code_sha256']=sha(Path(__file__))
        model=LocalModel().model
        for parameter in model.parameters():parameter.requires_grad=False
        model.eval()
        train_x,train_y=extract(model,train,'training',report)
        dev_x,dev_y=extract(model,dev,'development',report)
        torch.save({'train_x':train_x,'train_y':train_y,'dev_x':dev_x,'dev_y':dev_y},RUN/'features.pt')
        head=model.classifier
        with torch.no_grad():report['baseline']=metrics(dev_y.numpy(),head(dev_x).numpy())
        for parameter in head.parameters():parameter.requires_grad=True
        optimizer=torch.optim.AdamW(head.parameters(),lr=0.0001,weight_decay=0.01)
        best_score=-1.;best_state=None;best_epoch=None
        for epoch in range(1,26):
            head.train();total_loss=0.;order=torch.randperm(len(train_y))
            for indices in order.split(32):
                optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(train_x[indices]),train_y[indices]);loss.backward();optimizer.step()
                total_loss+=loss.item()*len(indices)
            head.eval()
            with torch.no_grad():measured=metrics(dev_y.numpy(),head(dev_x).numpy())
            report['history'].append({'epoch':epoch,'train_loss':total_loss/len(train_y),'development_macro_f1':measured['macro_f1'],'development_accuracy':measured['accuracy']})
            if measured['macro_f1']>best_score:
                best_score=measured['macro_f1'];best_state=copy.deepcopy(head.state_dict());best_epoch=epoch;report['candidate']=measured
            report['stage']=f'Training epoch {epoch}/25';write_report(report)
        head.load_state_dict(best_state)
        torch.save({'classifier_state_dict':best_state,'base_model_sha256':report['base_model_sha256'],'labels':LABELS,'run_id':report['run_id'],'epoch':best_epoch},RUN/'candidate-head.pt')
        with torch.no_grad():predictions=head(dev_x).softmax(-1).tolist()
        (RUN/'development-predictions.json').write_text(json.dumps([{'file':item['row']['audio']['path'],'truth':LABELS[item['label']],'scores':dict(zip(LABELS,scores))} for item,scores in zip(dev,predictions)],indent=2))
        report.update(status='complete',stage='Pilot training complete',best_epoch=best_epoch,
            candidate_sha256=sha(RUN/'candidate-head.pt'),completed_at=datetime.now(timezone.utc).isoformat(),
            decision='Candidate saved for further evaluation; original model remains active. Development results alone do not justify a deployment claim.')
        write_report(report)
    except Exception as error:
        report.update(status='failed',stage='Experiment failed',error=f'{type(error).__name__}: {error}');write_report(report);raise

if __name__=='__main__':main()
