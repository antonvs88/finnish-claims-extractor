"""Download authenticated GitHub release assets, verify, and extract locally."""
import subprocess,hashlib,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent;A=R/'release-assets';A.mkdir(exist_ok=True)
subprocess.run(['gh','release','download','research-2026-09-30','--repo','antonvs88/finnish-claims-extractor','--dir',str(A),'--pattern','models-*.tar.gz','--pattern','release-sha256.txt','--skip-existing'],check=True)
for line in (A/'release-sha256.txt').read_text().splitlines():
 digest,name=line.split();assert Path(name).name==name;p=A/name;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 assert h.hexdigest()==digest,f'Archive checksum mismatch: {name}'
 with tarfile.open(p,'r:gz') as archive:archive.extractall(R,filter='data')
subprocess.run([__import__('sys').executable,str(R/'verify_models.py')],check=True)
