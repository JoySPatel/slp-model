import json
from pathlib import Path
import tempfile
import unittest
from speech_feedback import analyze, load_input, render_report

class FeedbackTests(unittest.TestCase):
    def test_fillers(self):
        self.assertEqual(analyze('Umbrella UM uh you know I mean')['filler_count'], 4)
        self.assertEqual(analyze('Who are you? Know this?')['filler_count'], 0)
    def test_repetition(self):
        result = analyze('I I I think so. So we we agree.')
        self.assertEqual([(r['word'], r['count']) for r in result['repetitions']], [('i', 3), ('we', 2)])
    def test_timing(self):
        result = analyze('hello world', [{'start': 2, 'end': 3}, {'start': 4, 'end': 6}])
        self.assertEqual(result['pauses'], [{'start': 3, 'end': 4, 'seconds': 1}])
        self.assertEqual(result['words_per_minute'], 30)
    def test_empty(self):
        result = analyze('')
        self.assertIsNone(result['pauses'])
        self.assertIsNone(result['words_per_minute'])
        self.assertEqual(result['fillers_per_100_words'], 0)
    def test_escape(self):
        self.assertNotIn('<script>alert', render_report(analyze('<script>alert(1)</script>')))
    def test_threshold(self):
        for threshold in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): analyze('hello', pause_threshold=threshold)
    def test_invalid_input(self):
        invalid = [[], {}, {'segments': [None]}, {'segments': [{'text': 'a', 'start': True, 'end': 3}]}, {'segments': [{'text': 'a', 'start': 4, 'end': 2}]}, {'segments': [{'text': 'a', 'start': 0, 'end': 3}, {'text': 'b', 'start': 2, 'end': 4}]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            for data in invalid:
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load_input(path)
    def test_sample(self):
        text, segments = load_input(Path(__file__).resolve().parents[1] / 'examples/practice.json')
        result = analyze(text, segments)
        self.assertEqual(result['filler_count'], 3)
        self.assertEqual(len(result['repetitions']), 1)
        self.assertEqual(len(result['pauses']), 2)

if __name__ == '__main__': unittest.main()
