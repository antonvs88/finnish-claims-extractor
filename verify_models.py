import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
for entry in json.loads((R/'model-manifest.json').read_text()):
 p=R/entry['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 assert h.hexdigest()==entry['sha256'],f'Missing or changed model file: {p}'
print('All model hashes match')
