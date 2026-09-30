"""Select neutral-class prior adjustment on dev only; reserved narratives unopened."""
import json,sys,time
from pathlib import Path
import torch
from inference import NLI
R=Path(__file__).resolve().parent;torch.set_num_threads(8)
def classify(probabilities,factor):
 result={}
 for f,chunks in probabilities.items():
  labels=[]
  for p in chunks:
   scores=p.copy();scores['neutral']*=factor;labels.append(max(scores,key=scores.get))
  ent='entailment' in labels;con='contradiction' in labels
  result[f]=True if ent and not con else False if con and not ent else None
 return result

def metrics(pred,rows):
 pairs=[(r['expected'][f],pred[r['id']][f]) for r in rows for f in pred[r['id']]]
 recalls={str(v):sum(p is v for g,p in pairs if g is v)/sum(g is v for g,p in pairs) for v in [True,False,None]}
 return {'macro':sum(recalls.values())/3,'recall':recalls,'unsupported':sum(g is None and p is not None for g,p in pairs)}
if __name__=='__main__':
 rows=json.loads((R.parent/'varied-training/dev.json').read_text());path=R/'calibration_dev_probabilities.json'
 if path.exists():records=json.loads(path.read_text())
 else:
  model=NLI(str(R/'finetuned/best'));records=[]
  for i,r in enumerate(rows):
   _,probs,chunks=model.predict(r['text']);records.append({'id':r['id'],'probabilities':probs,'chunks':chunks})
   if i%24==0:print('dev rows',i,flush=True)
  path.write_text(json.dumps(records,ensure_ascii=False))
 grid=[]
 for factor in [1,2,4,8,16,32,64,128,256]:
  pred={r['id']:classify(r['probabilities'],factor) for r in records};s=metrics(pred,rows);grid.append({'neutral_factor':factor,**s})
 best=max(grid,key=lambda s:s['macro']);(R/'calibration.json').write_text(json.dumps({'selection':'maximum macro true/false/null recall on all24boolean fields of144dev documents','grid':grid,'best':best,'narrative_test_used':False},indent=2));print('Selected',best,flush=True)
 from score import evaluate
 probes=json.loads((R.parent/'varied-training/test.json').read_text());raw=json.loads((R/'finetuned_full_predictions.json').read_text());out=[]
 for r in raw:
  p=r['prediction'].copy();p.update(classify(r['boolean_probabilities'],best['neutral_factor']));out.append({'id':r['id'],'prediction':p})
 (R/'calibrated_full_predictions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));scores=evaluate(out,probes);(R/'calibrated_full_scores.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2));print(scores['known_value_accuracy'],scores['unsupported_values'],scores['exact_claims'],flush=True)
