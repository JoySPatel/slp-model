'use strict';
const $ = id => document.getElementById(id);
const prompts = {free:'Talk about anything you like.',day:'Tell yourself about a small moment from today. What stood out?',interest:'Talk about something you enjoy. What would you want someone to know about it?',intro:'Introduce yourself in whatever way feels comfortable. There’s no rush.',order:'Imagine ordering your favorite drink or meal. Take all the time you need.'};
const choices = ['Block','Sound / syllable repetition','Whole-word repetition','Repetition','Prolongation','Interjection','Ordinary pause','No event noticed','Something else'];
let recorder, stream, chunks = [], timer, started, session = null, buffer = null, url = null, dirty = false, analyzing = false, cancelled = false, modelReady = false, replayEnd = null, busy = false;
let dbPromise = new Promise((resolve, reject) => {const request = indexedDB.open('room-to-speak',1);request.onupgradeneeded = () => request.result.createObjectStore('sessions',{keyPath:'id'});request.onsuccess = () => resolve(request.result);request.onerror = () => reject(request.error);});
dbPromise.catch(() => notify('Browser storage is unavailable. You can still record, review, and download your audio.'));
async function storage(action, value) {const db = await dbPromise;return new Promise((resolve,reject)=>{const tx=db.transaction('sessions',action==='getAll'?'readonly':'readwrite');const req=tx.objectStore('sessions')[action](value);tx.oncomplete=()=>resolve(req.result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error);});}
function notify(text) {$('message').textContent=text;clearTimeout(notify.timer);notify.timer=setTimeout(()=>$('message').textContent='',7000);}
function time(seconds) {return `${Math.floor(seconds/60).toString().padStart(2,'0')}:${Math.floor(seconds%60).toString().padStart(2,'0')}`;}
function safeAction(fn) {return async (...args)=>{try{await fn(...args);}catch(error){notify(error.message || 'Something went wrong. Please try again.');}};}
function canReplace(){return !dirty || confirm('Discard the unsaved changes in this session?');}
function syncControls(){const recording=recorder?.state==='recording';$('record').disabled=recording||analyzing||busy;$('upload').disabled=recording||analyzing||busy;$('analyze').disabled=!modelReady||analyzing||recording||busy;$('save').disabled=analyzing||recording||busy;$('discard').disabled=analyzing||recording||busy;}
$('prompt').onchange=()=>{$('promptText').textContent=prompts[$('prompt').value];};
$('another').onclick=()=>{$('prompt').selectedIndex=($('prompt').selectedIndex+1)%$('prompt').options.length;$('prompt').onchange();};
async function status(){try {const response=await fetch('/api/status');const state=await response.json();modelReady=state.status==='ready';$('modelStatus').textContent=state.message;if(state.status==='loading')setTimeout(status,2000);syncControls();}catch{$('modelStatus').textContent='Local server disconnected. Restart the app to analyze audio.';modelReady=false;syncControls();}}
status();
function release(){clearInterval(timer);if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;$('record').hidden=false;$('stop').hidden=true;$('recordVisual').classList.remove('active');$('recordState').textContent='Ready to record';syncControls();}
$('record').onclick=safeAction(async()=>{
 if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder)throw Error('Recording isn’t supported here. Open this app in Chrome or Safari, or import an audio file.');
 if(!canReplace())return;
 busy=true;syncControls();
 try {
  stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
  const mime=['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(type=>MediaRecorder.isTypeSupported(type));
  recorder=new MediaRecorder(stream,mime?{mimeType:mime}:{});chunks=[];
  recorder.ondataavailable=event=>{if(event.data.size)chunks.push(event.data);};
  recorder.onerror=()=>{release();notify('Recording stopped because of a microphone error.');};
  recorder.onstop=()=>{const blob=new Blob(chunks,{type:recorder.mimeType});chunks=[];release();safeAction(()=>openBlob(blob))();};
  recorder.start(1000);started=Date.now();$('elapsed').textContent='00:00';timer=setInterval(()=>$('elapsed').textContent=time((Date.now()-started)/1000),500);
  $('record').hidden=true;$('stop').hidden=false;$('recordState').textContent='Recording…';$('recordVisual').classList.add('active');busy=false;syncControls();
 }catch(error){busy=false;release();throw Error(error.name==='NotAllowedError'?'Microphone permission was denied. Allow microphone access in your browser and try again.':error.message);}
});
$('stop').onclick=()=>{if(recorder?.state==='recording'){recorder.stop();$('stop').disabled=true;setTimeout(()=>$('stop').disabled=false,300);}};
$('upload').onchange=safeAction(async event=>{const file=event.target.files[0];event.target.value='';if(!file||!canReplace())return;if(file.size>100*1024*1024)throw Error('Please choose an audio file smaller than 100 MB.');await openBlob(file);});
async function decode(blob){const context=new AudioContext();try{return await context.decodeAudioData(await blob.arrayBuffer());}catch{throw Error('This audio format could not be opened. Try a WAV, MP3, or a new recording.');}finally{await context.close();}}
async function openBlob(blob, saved=null){
 busy=true;syncControls();
 try {
 const decoded=await decode(blob);if(decoded.duration<0.1)throw Error('The recording is too short to review. Please record a little longer.');
 buffer=decoded;session=saved||{id:crypto.randomUUID(),created:new Date().toISOString(),blob,events:[],title:'',comfort:'',notes:'',prompt:$('prompt').value,analyzed:false};
 if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(blob);$('player').src=url;replayEnd=null;
 $('review').hidden=false;$('reflection').hidden=false;$('title').value=session.title;$('comfort').value=session.comfort;$('notes').value=session.notes;
 $('end').value=Math.min(1,buffer.duration).toFixed(1);$('start').value='0';$('elapsed').textContent=time(buffer.duration);
 $('analysisStatus').textContent=session.analysisSummary||'Analysis is optional. You can add observations without running the model.';
 dirty=!saved;draw();renderEvents();
 } finally {busy=false;syncControls();}
}
function draw(){const canvas=$('wave'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!buffer)return;const samples=buffer.getChannelData(0),step=Math.max(1,Math.floor(samples.length/250));ctx.fillStyle='#91a681';for(let i=0;i<250;i++){let peak=0;for(let j=i*step;j<Math.min((i+1)*step,samples.length);j++)peak=Math.max(peak,Math.abs(samples[j]));const height=Math.max(2,peak*110);ctx.fillRect(i*4,65-height/2,2,height);}const x=($('player').currentTime/buffer.duration)*canvas.width;ctx.fillStyle='#355c47';ctx.fillRect(x,0,2,130);}
$('player').ontimeupdate=()=>{draw();if(replayEnd!==null&&$('player').currentTime>=replayEnd){$('player').pause();replayEnd=null;}};
$('player').onseeked=()=>draw();
async function replay(event){$('player').currentTime=event.start;replayEnd=event.end;await $('player').play();}
function element(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function renderEvents(){const container=$('events');container.replaceChildren();if(!session.events.length){container.append(element('p','No observations yet. Listen back, add your own, or try the experimental analysis.','small'));return;}
 [...session.events].sort((a,b)=>a.start-b.start).forEach(event=>{
 const card=element('div',undefined,'event'+(event.dismissed?' dismissed':''));const top=element('div',undefined,'event-top');
 const name=event.reviewed?event.label:(event.label==='Quiet audio'?'Quiet audio to review':`Possible ${event.label.toLowerCase()}`);
 top.append(element('span',`${event.source==='model'&&!event.reviewed?'◌':'●'} ${name}`,'event-name'));const play=element('button',`${time(event.start)}–${time(event.end)} ▶`);play.onclick=safeAction(()=>replay(event));top.append(play);card.append(top);
 card.append(element('p',event.source==='model'?(event.reviewed?'Reviewed by you · original model label: '+event.originalLabel:'Model suggestion · approximate 3-second window'):'Your observation','small'));
 if(event.label==='Block'&&!event.reviewed)card.append(element('p','A pause alone cannot establish a block. Did this moment feel stuck to you?','small'));
 if(event.label==='Quiet audio')card.append(element('p','Low audio level; this could be a pause or a quiet recording, not necessarily a block.','small'));
 const controls=element('div',undefined,'event-controls');const select=element('select');select.setAttribute('aria-label',`Your label for ${time(event.start)}`);const placeholder=element('option','Choose your observation');placeholder.value='';select.append(placeholder);for(const label of choices){const option=element('option',label);option.value=label;select.append(option);}select.value=event.reviewed?event.label:'';
 select.onchange=()=>{if(!select.value)return;event.label=select.value;event.reviewed=true;event.dismissed=false;dirty=true;renderEvents();};controls.append(select);
 const dismiss=element('button',event.dismissed?'Restore':'Dismiss');dismiss.onclick=()=>{event.dismissed=!event.dismissed;dirty=true;renderEvents();};controls.append(dismiss);
 if(event.source==='manual'){const remove=element('button','Remove');remove.onclick=()=>{session.events=session.events.filter(item=>item.id!==event.id);dirty=true;renderEvents();};controls.append(remove);}card.append(controls);
 if(event.scores&&Object.keys(event.scores).length){const detail=element('details');detail.append(element('summary','Model scores (not certainty)'));detail.append(element('p',Object.entries(event.scores).map(([label,score])=>`${label}: ${Math.round(score*100)}%`).join(' · '),'small'));card.append(detail);}container.append(card);
 });
}
$('here').onclick=()=>{if(!buffer)return;const start=Math.min($('player').currentTime,Math.max(0,buffer.duration-.1));$('start').value=start.toFixed(2);$('end').value=Math.min(start+1,buffer.duration).toFixed(2);};
$('add').onclick=()=>{if(!session)return;const start=Number($('start').value),end=Number($('end').value);if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>buffer.duration+.01)return notify('Choose a start and end within your recording, with end after start.');session.events.push({id:crypto.randomUUID(),start,end:Math.min(end,buffer.duration),label:$('kind').value,source:'manual',reviewed:true});dirty=true;renderEvents();};
$('analyze').onclick=safeAction(async()=>{
 if(!buffer||analyzing)return;
 if(session.events.some(event=>event.source==='model')&&!confirm('Replace prior model suggestions? Your reviewed and manually added observations will be kept.'))return;
 const chosenModel=$('analysisModel').value;
 analyzing=true;cancelled=false;$('cancel').hidden=false;syncControls();
 try{
  const offline=new OfflineAudioContext(1,Math.ceil(buffer.duration*16000),16000);const source=offline.createBufferSource();source.buffer=buffer;source.connect(offline.destination);source.start();const audio=(await offline.startRendering()).getChannelData(0);
  session.events=session.events.filter(event=>event.source!=='model'||event.reviewed);let completed=0,unflagged=0;const total=Math.ceil(audio.length/48000);
  for(let offset=0;offset<audio.length;offset+=48000){
   if(cancelled)break;const clip=audio.slice(offset,Math.min(offset+48000,audio.length));if(clip.length<1600)continue;
   $('analysisStatus').textContent=`Reviewing window ${completed+1} of ${total} on your device…`;
   const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Practice-App':'1','X-Speech-Model':chosenModel},body:clip.buffer});const result=await response.json();if(!response.ok)throw Error(result.error||'Analysis failed.');
   completed++;if(result.label==='Fluent')unflagged++;else session.events.push({id:crypto.randomUUID(),start:offset/16000,end:Math.min((offset+48000)/16000,buffer.duration),label:result.label,originalLabel:result.label,source:'model',reviewed:false,scores:result.scores});dirty=true;renderEvents();
  }
  session.analyzed=!cancelled;session.analysisSummary=`${chosenModel==='personal'?'Personal':'Original'} model. ${cancelled?'Stopped early. ':''}Reviewed ${completed} windows; ${unflagged} had no model flag. This does not mean no stuttering occurred. You can add missed moments.`;$('analysisStatus').textContent=session.analysisSummary;
 }catch(error){session.analyzed=false;session.analysisSummary='Analysis incomplete. Any results shown are partial. You can retry or add observations yourself.';$('analysisStatus').textContent=session.analysisSummary;throw error;}
 finally{analyzing=false;$('cancel').hidden=true;syncControls();}
});
$('cancel').onclick=()=>{cancelled=true;$('analysisStatus').textContent='Stopping after the current window…';};
for(const id of ['title','comfort','notes'])$(id).oninput=()=>dirty=true;
$('save').onclick=safeAction(async()=>{if(!session)return;session.title=$('title').value.trim()||'Practice session';session.comfort=$('comfort').value;session.notes=$('notes').value;await storage('put',session);dirty=false;notify('Saved in this browser on your device.');await listSessions();});
function reset(){if(url)URL.revokeObjectURL(url);url=null;session=null;buffer=null;dirty=false;$('player').pause();$('player').removeAttribute('src');$('player').load();$('review').hidden=true;$('reflection').hidden=true;$('elapsed').textContent='00:00';}
$('discard').onclick=()=>{if(!canReplace())return;reset();notify('Current session closed. Previously saved copies remain in Your sessions.');};
$('download').onclick=()=>{if(!session)return;const a=document.createElement('a');a.href=url;const type=session.blob.type;const ext=type.includes('mp4')?'m4a':type.includes('wav')?'wav':type.includes('mpeg')?'mp3':type.includes('ogg')?'ogg':'webm';a.download=`speech-practice-${session.created.slice(0,10)}.${ext}`;a.click();};
async function listSessions(){const sessions=await storage('getAll');const container=$('sessions');container.replaceChildren();if(!sessions.length){container.append(element('p','Your saved practice sessions will appear here.','small'));return;}
 sessions.sort((a,b)=>b.created.localeCompare(a.created)).forEach(saved=>{const row=element('div',undefined,'session');const open=element('button',saved.title);open.onclick=safeAction(async()=>{if(analyzing||busy||recorder?.state==='recording')return notify('Finish recording or analysis before opening another session.');if(canReplace())await openBlob(saved.blob,saved);});row.append(open,element('p',`${new Date(saved.created).toLocaleDateString()}${saved.comfort?' · '+saved.comfort:''}`,'small'));const remove=element('button','Delete','delete danger');remove.onclick=safeAction(async()=>{if(!confirm('Delete this saved recording and its notes from this browser?'))return;await storage('delete',saved.id);if(session?.id===saved.id)dirty=true;await listSessions();notify('Saved copy deleted. If open, discard the current session to clear it too.');});row.append(remove);container.append(row);});}
safeAction(listSessions)();
window.addEventListener('beforeunload',event=>{if(dirty||recorder?.state==='recording'||analyzing){event.preventDefault();event.returnValue='';}});

// Read-only experiment log generated by train_model.py. No simulated metrics.
async function experimentLog(){
 try {
  const response=await fetch('/api/experiment');if(!response.ok)throw Error('Experiment log unavailable.');
  const run=await response.json();$('trainingStatus').textContent=run.stage;
  const container=$('trainingDetails');container.replaceChildren();
  if(run.dataset){
   const source=element('p');const link=element('a',run.dataset);link.href='https://huggingface.co/datasets/psusac/sep28k';link.target='_blank';link.rel='noreferrer';source.append(link);container.append(source);
   container.append(element('p',run.license,'small'),element('p',run.method,'small'));
  }
  if(run.train){container.append(element('p',`${run.train.selected} training clips · ${run.development.selected} development clips · 0 shared speaker IDs or identical audio hashes in this run`,'small'));}
  if(run.baseline&&run.candidate){
   const table=element('table');table.className='metrics';const caption=element('caption','Development-set results (not an independent test)');table.append(caption);
   const header=element('tr');for(const title of ['Metric','Before','After training'])header.append(element('th',title));const thead=element('thead');thead.append(header);table.append(thead);const body=element('tbody');
   const percent=value=>(value*100).toFixed(1)+'%';
   for(const [name,before,after] of [['Macro F1',run.baseline.macro_f1,run.candidate.macro_f1],['Accuracy',run.baseline.accuracy,run.candidate.accuracy],...Object.keys(run.baseline.per_class).map(name=>[name+' F1',run.baseline.per_class[name].f1,run.candidate.per_class[name].f1])]){
    const row=element('tr');for(const value of [name,percent(before),percent(after)])row.append(element('td',value));body.append(row);
   }
   table.append(body);container.append(table,element('p','F1 balances missed events and incorrect flags. Macro F1 gives each class equal weight.','small'));
  }
  if(run.best_epoch)container.append(element('p',`Selected epoch ${run.best_epoch} of ${run.config.epochs}, using development macro F1.`,'small'));
  if(run.decision)container.append(element('p',run.decision,'small'));
  if(run.active_model)container.append(element('p','Default model: '+run.active_model,'small'));
  if(run.limitations){const details=element('details');details.open=true;details.append(element('summary','How to interpret this run'));const list=element('ul');list.className='small';for(const text of run.limitations)list.append(element('li',text));details.append(list);container.append(details);}
  if(run.error)container.append(element('p',run.error,'small'));
  if(run.updated_at)container.append(element('p',`${run.run_id} · Updated ${new Date(run.updated_at).toLocaleString()}`,'small'));
  const history=$('trainingHistory');history.replaceChildren();
  if(run.history?.length){const table=element('table');table.className='metrics';const header=element('tr');for(const label of ['Epoch','Training loss','Dev macro F1'])header.append(element('th',label));table.append(header);for(const entry of run.history){const row=element('tr');for(const value of [entry.epoch,entry.train_loss.toFixed(4),(entry.development_macro_f1*100).toFixed(1)+'%'])row.append(element('td',value));table.append(row);}history.append(table);}
  else history.append(element('p','No completed training epochs yet.','small'));
  if(run.status==='running')setTimeout(experimentLog,5000);
 }catch{$('trainingStatus').textContent='Experiment log unavailable. Restart the local app and refresh to load it.';}
}
experimentLog();
