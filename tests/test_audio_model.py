"""Model integration checks, not a clinical accuracy evaluation."""
import unittest
import numpy as np
from model import LocalModel

class AudioModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model = LocalModel()
    def test_silence_is_not_called_a_block(self):
        self.assertEqual(self.model.predict(np.zeros(48000))['label'], 'Quiet audio')
    def test_reject_invalid_pcm(self):
        for audio in (np.zeros(10), np.zeros(48001), np.full(48000, np.nan), np.full(48000, 2)):
            with self.assertRaises(ValueError): self.model.predict(audio)
    def test_inference_is_finite_and_input_dependent(self):
        t = np.arange(48000, dtype=np.float32)/16000
        first = self.model.predict(.2*np.sin(2*np.pi*220*t))
        second = self.model.predict(np.random.default_rng(42).normal(0,.1,48000))
        self.assertEqual(set(first['scores']), {'Prolongation','Repetition','Block','Interjection','Fluent'})
        self.assertAlmostEqual(sum(first['scores'].values()), 1, places=5)
        self.assertTrue(all(np.isfinite(list(first['scores'].values()))))
        self.assertFalse(np.allclose(list(first['scores'].values()),list(second['scores'].values()),atol=.001))

if __name__ == '__main__': unittest.main()
