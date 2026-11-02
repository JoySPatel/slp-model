import base64
import tempfile
from pathlib import Path
import unittest
import numpy as np
import personal_training as pt
from model import LocalModel

class PersonalTrainingTests(unittest.TestCase):
    def payload(self):
        rng=np.random.default_rng(12)
        recordings=[]
        for i in range(3):
            audio=rng.normal(0,.03,48000*5).astype('<f4')
            recordings.append({'role':'evaluation' if i==2 else 'train','labels':['Fluent','Repetition','Fluent','Repetition','Fluent'],'pcm':base64.b64encode(audio.tobytes()).decode()})
        return {'consent':True,'recordings':recordings}
    def test_valid_whole_recording_split(self):
        clips,summary=pt.validate(self.payload())
        self.assertEqual(len(clips),15)
        self.assertEqual(summary['recordings'],{'train':2,'evaluation':1})
    def test_allows_evaluation_only_class_and_reports_it(self):
        data=self.payload()
        data['recordings'][2]['labels'][0]='Block'
        clips,summary=pt.validate(data)
        self.assertEqual(summary['evaluation_only_classes'],['Block'])
        self.assertEqual(summary['clips']['evaluation'],5)
        self.assertTrue(any(role=='evaluation' and label==2 for role,_,label in clips))
    def test_requires_consent_and_review(self):
        data=self.payload();data['consent']=False
        with self.assertRaises(ValueError):pt.validate(data)
        data['consent']=True;data['recordings'][0]['labels'][0]=None
        with self.assertRaises(ValueError):pt.validate(data)
    def test_rejects_duplicate_audio_and_missing_split(self):
        data=self.payload();data['recordings'][2]['pcm']=data['recordings'][0]['pcm']
        with self.assertRaises(ValueError):pt.validate(data)
        data=self.payload();data['recordings'][2]['role']='train'
        with self.assertRaises(ValueError):pt.validate(data)
    def test_training_writes_candidate_and_can_infer(self):
        clips,summary=pt.validate(self.payload());old_directory=pt.DIRECTORY;old_status=pt.STATUS;old_head=pt.HEAD
        try:
            with tempfile.TemporaryDirectory() as folder:
                pt.DIRECTORY=Path(folder);report={'status':'running','stage':'Testing','run_id':'synthetic-test','data':summary,'history':[]}
                pt.GUARD.acquire();pt.worker(clips,report)
                self.assertEqual(pt.STATUS['status'],'complete',pt.STATUS)
                self.assertEqual(len(pt.STATUS['history']),15)
                base=LocalModel();head=pt.personal_head(base.model)
                output=base.predict(clips[0][1],head=head)
                self.assertEqual(len(output['scores']),5)
                self.assertAlmostEqual(sum(output['scores'].values()),1,places=5)
        finally:pt.DIRECTORY=old_directory;pt.STATUS=old_status;pt.HEAD=old_head

if __name__=='__main__':unittest.main()
