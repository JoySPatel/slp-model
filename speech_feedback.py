"""Offline, standard-library speech practice analysis. Python 3.10+."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import re

TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
FILLERS = {('um',), ('uh',), ('erm',), ('hmm',), ('you', 'know'), ('i', 'mean')}


def load_input(path: Path) -> tuple[str, list[dict] | None]:
    """Read plain text or ordered, non-overlapping timed segments in seconds."""
    if path.suffix.lower() != '.json':
        return path.read_text(encoding='utf-8').strip(), None
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('segments'), list):
        raise ValueError('JSON must contain a segments list.')
    segments = data['segments']
    previous_end = 0.0
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get('text'), str):
            raise ValueError('Every segment needs text, start, and end.')
        start, end = segment.get('start'), segment.get('end')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)):
            raise ValueError('Segment timestamps must be finite numbers in seconds.')
        if start < previous_end or end <= start:
            raise ValueError('Segments must be ordered, non-overlapping, and have end > start >= 0.')
        previous_end = end
    return ' '.join(s['text'].strip() for s in segments), segments


def analyze(text: str, segments: list[dict] | None = None, pause_threshold: float = 1.0) -> dict:
    if not math.isfinite(pause_threshold) or pause_threshold <= 0:
        raise ValueError('Pause threshold must be a positive finite number.')
    tokens = list(TOKEN.finditer(text))
    words = [m.group().lower().replace('’', "'") for m in tokens]
    fillers = []
    i = 0
    while i < len(words):
        phrase = next((p for p in sorted(FILLERS, key=lambda p: (-len(p), p))
                       if tuple(words[i:i + len(p)]) == p
                       and not re.search(r'[.!?;:\n]', text[tokens[i].start():tokens[i + len(p) - 1].end()])), None)
        if phrase:
            fillers.append({'phrase': ' '.join(phrase), 'word_position': i + 1})
            i += len(phrase)
        else:
            i += 1
    repetitions = []
    i = 0
    while i < len(words):
        end = i + 1
        while (end < len(words) and words[end] == words[i]
               and not re.search(r'[.!?;:\n]', text[tokens[end - 1].end():tokens[end].start()])):
            end += 1
        if end - i > 1:
            repetitions.append({'word': words[i], 'count': end - i, 'word_position': i + 1})
        i = end
    pauses = []
    if segments is not None:
        for left, right in zip(segments, segments[1:]):
            gap = right['start'] - left['end']
            if gap >= pause_threshold:
                pauses.append({'start': left['end'], 'end': right['start'], 'seconds': round(gap, 3)})
    duration = segments[-1]['end'] - segments[0]['start'] if segments else None
    feedback = []
    if not words:
        feedback.append('Add a transcript with English words to get practice feedback.')
    else:
        if fillers:
            common = Counter(f['phrase'] for f in fillers).most_common(1)[0][0]
            feedback.append(f'Listen for “{common}” on your next take. Try replacing one occurrence with a deliberate breath.')
        if repetitions:
            feedback.append('Review the repeated-word examples. Practice restarting a phrase once, slowly; intentional repetition is fine.')
        if pauses:
            feedback.append('Review the timed gaps below. Keep pauses that help your audience; rehearse transitions that felt unplanned.')
        if not fillers and not repetitions:
            feedback.append('No listed fillers or adjacent repeated words were detected. Practice clear emphasis and transitions next.')
    if segments is None:
        feedback.append('Pause timing and speaking rate need timestamped segments; punctuation alone cannot measure silence.')
    return {'word_count': len(words), 'filler_count': len(fillers),
            'fillers_per_100_words': round(len(fillers) * 100 / len(words), 2) if words else 0,
            'fillers': fillers, 'repetitions': repetitions,
            'pause_threshold_seconds': pause_threshold, 'pauses': pauses if segments is not None else None,
            'duration_seconds': round(duration, 3) if duration is not None else None,
            'words_per_minute': round(len(words) * 60 / duration, 1) if duration else None,
            'feedback': feedback, 'transcript': text}


def render_report(result: dict) -> str:
    escape = lambda value: html.escape(str(value))
    cards = [('Words', result['word_count']), ('Potential fillers', result['filler_count']),
             ('Repeated-word runs', len(result['repetitions'])),
             ('Timed pauses', len(result['pauses']) if result['pauses'] is not None else 'N/A'),
             ('Words / minute', result['words_per_minute'] if result['words_per_minute'] is not None else 'N/A')]
    stats = ''.join(f'<div class="card"><strong>{escape(value)}</strong><span>{label}</span></div>' for label, value in cards)
    feedback = ''.join(f'<li>{escape(item)}</li>' for item in result['feedback'])
    rows = []
    for item in result['fillers']:
        rows.append(('Potential filler', item['phrase'], f"Word {item['word_position']}"))
    for item in result['repetitions']:
        rows.append(('Repeated word', f"{item['word']} × {item['count']}", f"Word {item['word_position']}"))
    for item in result['pauses'] or []:
        rows.append(('Timed gap', f"{item['seconds']} seconds", f"{item['start']}–{item['end']} s"))
    evidence = ''.join('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>' for row in rows)
    if not evidence:
        evidence = '<tr><td colspan="3">No matching patterns detected.</td></tr>'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech Feedback · Practice report</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f4;color:#18382e;font:16px/1.6 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:60px 24px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#4b6a5c}h1{font-size:clamp(32px,6vw,54px);line-height:1.1;margin:16px 0}h2{font-size:22px}p{color:#50665c}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:32px 0}.card{background:#183e32;color:white;padding:22px;border-radius:16px}.card strong{display:block;font-size:34px}.card span{font-size:13px;color:#cce3d7}section{background:white;border:1px solid #dce5dd;border-radius:18px;padding:24px;margin:18px 0}li{margin:12px 0}.table{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px;border-bottom:1px solid #e6ece7}th{font-size:12px;text-transform:uppercase;letter-spacing:1px}.transcript{white-space:pre-wrap}.note{font-size:13px}footer{margin-top:28px;font-size:13px;color:#50665c}@media print{main{padding:0}.card{border:1px solid #aaa}section{break-inside:avoid}}</style><main>
<div class="eyebrow">Speech feedback / Python prototype</div><h1>A clearer next take.</h1><p>Patterns to review, with specific suggestions for your next practice session.</p>
''' + f'''<div class="stats">{stats}</div><section><h2>Your next practice session</h2><ul>{feedback}</ul></section>
<section><h2>Pattern details</h2><p class="note">Potential filler occurrences per 100 words: {result['fillers_per_100_words']}. Pause threshold: {result['pause_threshold_seconds']} seconds.</p><div class="table"><table><thead><tr><th>Pattern</th><th>Observed</th><th>Location</th></tr></thead><tbody>{evidence}</tbody></table></div></section>
<section><h2>Transcript</h2><div class="transcript">{escape(result['transcript'])}</div></section>
<footer>Rule-based English transcript analysis. Fillers and repetition can be intentional. Gaps come from supplied timestamps, not audio verification. Rate includes gaps between the first and last segments. No audio recognition, diagnosis, or overall quality score. Runs locally with no network requests.</footer></main></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='UTF-8 transcript .txt or timestamped .json')
    parser.add_argument('--pause-threshold', type=float, default=1.0, help='Minimum gap in seconds (default: 1)')
    parser.add_argument('--html', type=Path, help='Write a standalone HTML report')
    parser.add_argument('--json', type=Path, help='Write structured JSON results')
    args = parser.parse_args()
    try:
        text, segments = load_input(args.input)
        result = analyze(text, segments, args.pause_threshold)
        if args.html:
            args.html.write_text(render_report(result), encoding='utf-8')
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Speech Feedback | {result['word_count']} words | {result['filler_count']} potential fillers | {len(result['repetitions'])} repeated-word runs")
    for suggestion in result['feedback']:
        print(f'  • {suggestion}')
    for path in (args.html, args.json):
        if path:
            print(f'Saved: {path.resolve()}')


if __name__ == '__main__':
    main()
