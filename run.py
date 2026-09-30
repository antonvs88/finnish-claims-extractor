"""Offline inference for the frozen research candidate (device: CLAIMS_EXTRACTOR_DEVICE, else mps, cuda or cpu)."""
import os,sys,json,argparse
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parent
for folder in ['training','nli-baseline','natural-v4','country-evidence-v7','enum-nli-v5','overnight-2026-09-29']:sys.path.insert(0,str(ROOT/'src'/folder))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();output=Path(args.output)
 if output.exists():ap.error('Output exists; choose a new path')
 rows=json.loads(Path(args.input).read_text());assert isinstance(rows,list) and rows and all(isinstance(r.get('text'),str) and r['text'].strip() for r in rows)
 import torch
 from schema_nli import SchemaNLI,HYP
 from country import CountryNLI,decode
 from calibrate import classify
 from local_role_model import Joint
 from inference import DEVICE
 import joint_model as jm
 from enum_chunk_decode import decode as enum_decode
 from ensemble import ProbabilityEnsemble,enum_values
 torch.set_num_threads(min(8,os.cpu_count() or 8));cfg=json.loads((ROOT/'config.json').read_text());path=lambda s:str(ROOT/s)
 geography=CountryNLI(path(cfg['boolean']['base_checkpoint']));geography.fields=[f for f in geography.fields if f!='damageInNordicArea'];geography.hypotheses=[HYP[f] for f in geography.fields]
 new=SchemaNLI(path(cfg['boolean']['checkpoint']),mode='boolean');new.fields=[f for f in new.fields if f!='damageInNordicArea'];new.hypotheses=[HYP[f] for f in new.fields]
 boolean=ProbabilityEnsemble(geography,new,cfg['boolean']['weight'],cfg['boolean']['method'])
 event=SchemaNLI(path(cfg['event']['checkpoint']),mode='enum');event.fields=[f for f in event.fields if f.startswith('damageCause::')];event.hypotheses=[HYP[f] for f in event.fields]
 enums=SchemaNLI(path(cfg['other_enum']['checkpoint']),mode='enum');enums.fields=[f for f in enums.fields if not f.startswith('damageCause::')];enums.hypotheses=[HYP[f] for f in enums.fields]
 for n in [geography,new,event]:
  if DEVICE!='cpu':n.model.half()   # the frozen half precision; CPU stays float32, where half is slow
  n.model.eval()
 values=Joint().to(DEVICE);values.load_state_dict(torch.load(path(cfg['amount']['checkpoint']),map_location=DEVICE,weights_only=True),strict=True);values.eval()
 texts=[r['text'] for r in rows]
 for t in texts:assert len(jm.t.TOK(t,truncation=False)['input_ids'])<=4096,'Input exceeds4096 tokens; no silent truncation'
 out=[]
 for i in range(0,len(texts),8):out.extend(jm.predict(values,texts[i:i+8],cfg['amount']['threshold'],cfg['v4_enum_null_factor']))
 bs=boolean.predict_many(texts,batch_size=16);es=enums.predict_many(texts,batch_size=16);evs=event.predict_many(texts,batch_size=16);gs=geography.predict_country(texts,batch_size=16)
 for p,b,e,ev,g in zip(out,bs,es,evs,gs):
  p.update(classify(b,cfg['boolean']['factor']))
  for f,cal in cfg['enum_field_calibration'].items():p.update(enum_decode({k:v for k,v in e.items() if k.split('::')[0]==f},cal['factor'],cal['margin'],'strongest_chunk'))
  p.update(enum_values(ev,cfg['event']['factor'],cfg['event']['margin']));p['damageInNordicArea']=decode(g,cfg['geography']['factor'],cfg['geography']['threshold'])
 from score import valid
 assert all(valid(p) for p in out)
 with output.open('x') as f:json.dump([{'id':r.get('id',str(i)),'prediction':p} for i,(r,p) in enumerate(zip(rows,out))],f,ensure_ascii=False,indent=2)
 print('Extracted',len(out),'claims')
if __name__=='__main__':main()
