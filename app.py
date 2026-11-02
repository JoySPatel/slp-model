"""Serve the private practice app on loopback; audio is processed in memory only."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATE = {'status': 'loading', 'message': 'Loading the local research model…'}
MODEL = None
LOCK = threading.Lock()

def load_model():
    global MODEL
    try:
        from model import LocalModel
        MODEL = LocalModel()
        STATE.update(status='ready', message='Experimental model ready · processing stays on this Mac')
    except Exception as error:
        STATE.update(status='unavailable', message='Model unavailable. Recording and manual review still work. See the terminal.')
        print(f'Model load failed: {type(error).__name__}: {error}', flush=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Do not log recording contents or request details.
    def respond(self, status, body, content_type='application/json'):
        if not isinstance(body, bytes): body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)
    def allowed(self):
        port = self.server.server_port
        return self.headers.get('Host') in {f'127.0.0.1:{port}', f'localhost:{port}'}
    def do_GET(self):
        if not self.allowed(): return self.respond(403, {'error': 'Loopback host required.'})
        path = urlparse(self.path).path
        if path == '/api/status': return self.respond(200, STATE)
        if path == '/api/personal':
            import personal_training
            return self.respond(200, personal_training.STATUS)
        if path == '/api/experiment':
            report = ROOT / 'web/training.json'
            if not report.exists(): return self.respond(200, {'status': 'not_started', 'stage': 'No training run yet'})
            return self.respond(200, report.read_bytes())
        files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css'), '/personal.js': ('personal.js', 'text/javascript')}
        if path not in files: return self.respond(404, {'error': 'Not found'})
        name, mime = files[path]
        self.respond(200, (ROOT / 'web' / name).read_bytes(), mime)
    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.allowed() or origin != f"http://{self.headers.get('Host')}" or self.headers.get('X-Practice-App') != '1':
            return self.respond(403, {'error': 'Same-origin local app requests only.'})
        if self.path == '/api/personal/train':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 1 <= length <= 55_000_000: raise ValueError('Training request is too large.')
                if self.headers.get('Content-Type') != 'application/json': raise ValueError('Expected JSON.')
                self.connection.settimeout(60)
                raw = self.rfile.read(length)
                if len(raw) != length: raise ValueError('Incomplete request.')
                import personal_training
                return self.respond(202, personal_training.start(json.loads(raw)))
            except (ValueError, TimeoutError) as error:
                return self.respond(400, {'error': str(error)})
        if self.path != '/api/analyze': return self.respond(404, {'error': 'Not found'})
        if MODEL is None: return self.respond(503, {'error': STATE['message']})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 6400 or length > 192000 or length % 4: raise ValueError('Invalid audio window size.')
            if self.headers.get('Content-Type') != 'application/octet-stream': raise ValueError('Expected PCM bytes.')
            self.connection.settimeout(20)
            raw = self.rfile.read(length)
            if len(raw) != length: raise ValueError('Incomplete audio window.')
            import numpy as np
            with LOCK:
                choice = self.headers.get('X-Speech-Model', 'original')
                if choice not in ('original', 'personal'): raise ValueError('Unknown model choice.')
                head = None
                if choice == 'personal':
                    import personal_training
                    head = personal_training.personal_head(MODEL.model)
                result = MODEL.predict(np.frombuffer(raw, dtype='<f4').copy(), head=head)
                result['model'] = choice
            self.respond(200, result)
        except (ValueError, TimeoutError) as error:
            self.respond(400, {'error': str(error)})
        except Exception:
            self.respond(500, {'error': 'Local analysis failed. You can still review and annotate your recording.'})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=load_model, daemon=True).start()
    print(f'Open http://127.0.0.1:{args.port} · Ctrl+C to stop', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
