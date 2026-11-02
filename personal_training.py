"""Opt-in personal head training. Audio stays in memory; only weights/report persist."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import threading
from datetime import datetime, timezone
import uuid
import numpy as np
import torch
from model import LocalModel, LABELS

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/'personal_models'
GUARD=threading.Lock()
STATUS={'status':'idle','stage':'No personal model trained yet.'}
HEAD=None

def restore():
    global STATUS
    path=DIRECTORY/'report.json'
    if path.exists():
        STATUS=json.loads(path.read_text())
        if STATUS.get('status')=='running': STATUS={'status':'interrupted','stage':'Previous training was interrupted. You can train again.'}
restore()

def publish(report):
    global STATUS
    report['updated_at']=datetime.now(timezone.utc).isoformat()
    DIRECTORY.mkdir(exist_ok=True)
    temp=DIRECTORY/'report.tmp';temp.write_text(json.dumps(report,allow_nan=False));temp.replace(DIRECTORY/'report.json')
    STATUS=copy.deepcopy(report)

def validate(payload):
    if not isinstance(payload,dict) or payload.get('consent') is not True:raise ValueError('Explicit training consent is required.')
    recordings=payload.get('recordings')
    if not isinstance(recordings,list) or not 3<=len(recordings)<=30:raise ValueError('Add at least two training recordings and one evaluation recording (maximum 30).')
    result=[];hashes=set();total=0;roles={'train':0,'evaluation':0};counts={'train':0,'evaluation':0};classes={'train':set(),'evaluation':set()};clip_hashes={}
    for recording in recordings:
        if not isinstance(recording,dict):raise ValueError('Invalid recording.')
        role=recording.get('role');labels=recording.get('labels');encoded=recording.get('pcm')
        if not isinstance(role,str) or role not in roles or not isinstance(labels,list) or not isinstance(encoded,str):raise ValueError('Invalid recording fields.')
        try:raw=base64.b64decode(encoded,validate=True)
        except Exception:raise ValueError('Invalid audio encoding.')
        if len(raw)%4 or not 6400<=len(raw)<=16000*4*300:raise ValueError('Each recording must be 0.1 seconds to 5 minutes.')
        total+=len(raw)
        if total>16000*4*600:raise ValueError('Use at most 10 minutes of audio per training run.')
        audio=np.frombuffer(raw,dtype='<f4').copy()
        if not np.isfinite(audio).all() or np.max(np.abs(audio))>1.01:raise ValueError('Invalid PCM samples.')
        digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes:raise ValueError('Duplicate recording detected. Use different takes for training and evaluation.')
        hashes.add(digest)
        windows=[audio[start:start+48000] for start in range(0,len(audio),48000) if len(audio[start:start+48000])>=1600]
        if len(labels)!=len(windows):raise ValueError('Clip labels do not match recording length.')
        used=0
        for clip,label in zip(windows,labels):
            if label is None:raise ValueError('Label every clip or explicitly mark it Skip.')
            if label=='skip':continue
            if not isinstance(label,str) or label not in LABELS:raise ValueError('Unknown clip label.')
            # Quantized samples catch equivalent PCM with small float encoding changes.
            clip_hash=hashlib.sha256((clip*32767).astype('<i2').tobytes()).hexdigest()
            if clip_hash in clip_hashes and clip_hashes[clip_hash]!=role:raise ValueError('Identical clips appear in training and evaluation. Use distinct recordings.')
            clip_hashes[clip_hash]=role
            result.append((role,clip,LABELS.index(label)));counts[role]+=1;classes[role].add(label);used+=1
        if used:roles[role]+=1
    if roles['train']<2 or roles['evaluation']<1:raise ValueError('Label clips from at least two training recordings and one separate evaluation recording.')
    if counts['train']<10 or counts['evaluation']<5:raise ValueError('Need at least 10 labeled training clips and 5 evaluation clips.')
    if len(classes['train'])<2:raise ValueError('Training needs at least two label types. Include speech with no event as well as stuttering when available.')
    return result,{'recordings':roles,'clips':counts,'training_classes':sorted(classes['train']),'evaluation_classes':sorted(classes['evaluation']),'evaluation_only_classes':sorted(classes['evaluation']-classes['train'])}

def measure(labels,logits):
    predicted=logits.argmax(1);per={}
    for index,name in enumerate(LABELS):
        tp=int(((labels==index)&(predicted==index)).sum());fp=int(((labels!=index)&(predicted==index)).sum());fn=int(((labels==index)&(predicted!=index)).sum());support=int((labels==index).sum())
        per[name]={'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'support':support}
    represented=[v['f1'] for v in per.values() if v['support']]
    return {'accuracy':float((labels==predicted).float().mean()),'macro_f1_observed_classes':sum(represented)/len(represented),'per_class':per}

def start(payload):
    if not GUARD.acquire(blocking=False):raise ValueError('Personal training is already running.')
    try:
        clips,summary=validate(payload)
        report={'status':'running','stage':'Loading model for personal training…','run_id':uuid.uuid4().hex,'data':summary,'epochs':15,'history':[],
                'note':'Fixed 15 epochs; evaluation recordings are scored only after training and never used to choose an epoch. Small self-labeled results are not clinical validation.'}
        publish(report)
        threading.Thread(target=worker,args=(clips,report),daemon=True).start()
        return report
    except Exception:GUARD.release();raise

def worker(clips,report):
    global HEAD
    try:
        torch.manual_seed(42)
        model=LocalModel().model
        for parameter in model.parameters():parameter.requires_grad=False
        xs={'train':[],'evaluation':[]};ys={'train':[],'evaluation':[]}
        for index,(role,audio,label) in enumerate(clips):
            padded=np.pad(audio,(0,48000-len(audio)))
            with torch.no_grad():feature=model.pooling(model.wavlm(torch.from_numpy(padded).unsqueeze(0)).last_hidden_state).detach()
            xs[role].append(feature);ys[role].append(label)
            report['stage']=f'Processing labeled clip {index+1}/{len(clips)}';publish(report)
        train_x=torch.cat(xs['train']);eval_x=torch.cat(xs['evaluation']);train_y=torch.tensor(ys['train']);eval_y=torch.tensor(ys['evaluation'])
        head=model.classifier
        with torch.no_grad():baseline=head(eval_x).detach().clone()
        for parameter in head.parameters():parameter.requires_grad=True
        optimizer=torch.optim.AdamW(head.parameters(),lr=0.0001,weight_decay=0.01)
        # Pull weights gently toward the original head to limit small-data drift.
        reference={key:value.detach().clone() for key,value in head.named_parameters()}
        for epoch in range(1,16):
            head.train();total_loss=0
            for batch in torch.randperm(len(train_y)).split(16):
                optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(head(train_x[batch]),train_y[batch])
                penalty=sum((value-reference[key]).square().mean() for key,value in head.named_parameters())
                (loss+0.1*penalty).backward();optimizer.step();total_loss+=float(loss.detach())*len(batch)
            report['history'].append({'epoch':epoch,'training_loss':total_loss/len(train_y)})
            report['stage']=f'Training epoch {epoch}/15';publish(report)
        head.eval()
        with torch.no_grad():report['baseline']=measure(eval_y,baseline);report['candidate']=measure(eval_y,head(eval_x))
        state={key:value.detach().cpu() for key,value in head.state_dict().items()}
        target=DIRECTORY/(report['run_id']+'.pt');torch.save({'classifier_state_dict':state,'labels':LABELS},target)
        HEAD=None
        report.update(status='complete',stage='Personal model trained. Review the results before choosing it.',checkpoint=target.name)
        publish(report)
    except Exception as error:
        report.update(status='failed',stage='Training failed; original model remains available.',error=str(error));publish(report)
    finally:GUARD.release()

def personal_head(base_model):
    global HEAD
    if STATUS.get('status')!='complete':raise ValueError('No completed personal model is available.')
    if HEAD is None:
        checkpoint=torch.load(DIRECTORY/STATUS['checkpoint'],map_location='cpu',weights_only=True)
        HEAD=copy.deepcopy(base_model.classifier)
        HEAD.load_state_dict(checkpoint['classifier_state_dict'],strict=True);HEAD.eval()
    return HEAD
