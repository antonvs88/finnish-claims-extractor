import json,math,statistics,collections,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;CAT=json.loads((R.parent/'input_catalog.json').read_text());FIELDS=list(CAT)
def correct(p,g):
 if g is None or type(g)==bool:return p is g
 if type(g) in (int,float):return type(p) in (int,float) and math.isfinite(p) and abs(p-g)<.005
 return type(p)==str and p==g

def valid(p):
 if set(p)!=set(FIELDS):return False
 for f,v in p.items():
  if v is None:continue
  t=CAT[f]['type']
  if t=='boolean' and type(v)!=bool:return False
  if t=='enum' and v not in CAT[f]['values']:return False
  if t=='number' and (type(v) not in (int,float) or not math.isfinite(v)):return False
  if t=='date':
   try:
    if datetime.date.fromisoformat(v).isoformat()!=v:return False
   except (ValueError,TypeError):return False
 return True

def evaluate(rows,gold):
 index={r['id']:r for r in rows};assert set(index)=={r['id'] for r in gold}
 cells=[];exact=0;errors=[];schema=0
 for c in gold:
  p=index[c['id']]['prediction'];schema+=valid(p);good=True
  for f,g in c['expected'].items():
   v=p.get(f);ok=f in p and correct(v,g);cells.append((CAT[f]['type'],g,v,ok));good &= ok
   if not ok:errors.append({'id':c['id'],'field':f,'gold':g,'prediction':v})
  exact+=good and valid(p)
 def rate(xs):return {'correct':sum(c[3] for c in xs),'total':len(xs),'rate':sum(c[3] for c in xs)/len(xs) if xs else None}
 result={'field_accuracy':rate(cells),'exact_claims':{'correct':exact,'total':len(gold)},'valid_schema':schema,'per_type':{t:rate([c for c in cells if c[0]==t]) for t in ['boolean','enum','number','date']},'known_value_per_type':{t:rate([c for c in cells if c[0]==t and c[1] is not None]) for t in ['boolean','enum','number','date']},'known_value_accuracy':rate([c for c in cells if c[1] is not None]),'null_accuracy':rate([c for c in cells if c[1] is None]),'answered_precision':rate([c for c in cells if c[2] is not None]),'unsupported_values':sum(g is None and p is not None for t,g,p,ok in cells),'boolean_confusion':dict(collections.Counter(str(g)+' -> '+str(p) for t,g,p,ok in cells if t=='boolean')),'errors':errors}
 if all('seconds' in r for r in rows):
  ts=[r['seconds'] for r in rows];result['seconds_per_claim']=statistics.mean(ts);result['claims_per_hour']=3600/statistics.mean(ts)
 return result
if __name__=='__main__':
 scores={}
 for split in ['test','challenge']:
  gold=json.loads((R/(split+'.json')).read_text())
  scores['all_null_'+split]=evaluate([{'id':r['id'],'prediction':dict.fromkeys(FIELDS)} for r in gold],gold)
  for model in ['modernbert','gemma']:
   path=R/(model+'_'+split+'.jsonl')
   if not path.exists():continue
   rows=[json.loads(x) for x in path.read_text().splitlines()]
   if len(rows)!=len(gold):continue
   s=evaluate(rows,gold);scores[model+'_'+split]=s;print(model,split,s['field_accuracy'],s['known_value_accuracy'],s['exact_claims'],round(s['claims_per_hour']),flush=True)
 (R/'scores.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2))
