import unittest
import numpy as np
from train_model import target, metrics

class TrainingTests(unittest.TestCase):
    def row(self, **changes):
        row=dict.fromkeys(['Unsure','PoorAudioQuality','DifficultToUnderstand','Music','NoSpeech','Prolongation','SoundRep','WordRep','Block','Interjection','NoStutteredWords'],0)
        return row|changes
    def test_ambiguous_labels_are_excluded(self):
        self.assertIsNone(target(self.row(Block=2,Prolongation=2)))
        self.assertIsNone(target(self.row(Block=2,NoStutteredWords=2)))
        self.assertIsNone(target(self.row(NoSpeech=2,NoStutteredWords=3)))
        self.assertIsNone(target(self.row()))
    def test_mapping(self):
        self.assertEqual(target(self.row(Block=2)),2)
        self.assertEqual(target(self.row(SoundRep=3)),1)
        self.assertEqual(target(self.row(WordRep=2)),1)
        self.assertEqual(target(self.row(NoStutteredWords=3)),4)
    def test_metrics(self):
        result=metrics([0,1,2,3,4],np.eye(5))
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['macro_f1'],1)
        wrong=metrics([0,1,2,3,4],np.roll(np.eye(5),1,axis=1))
        self.assertEqual(wrong['macro_f1'],0)
        self.assertEqual(wrong['accuracy'],0)
    def test_missing_predictions_have_zero_f1(self):
        result=metrics([0,1,2,3,4],np.tile([1,0,0,0,0],(5,1)))
        self.assertEqual(result['per_class']['Block']['f1'],0)
        self.assertEqual(result['per_class']['Block']['support'],1)

if __name__=='__main__':unittest.main()
