"""All-field boolean NLI inference. Long inputs split by lines/token windows, no silent truncation."""
import json,os,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'training'))
from build_data import B
DEVICE=os.environ.get('CLAIMS_EXTRACTOR_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
class NLI:
 def __init__(self,path=None):
  assert path is not None;self.path=path;self.tok=AutoTokenizer.from_pretrained(self.path);self.model=AutoModelForSequenceClassification.from_pretrained(self.path).to(DEVICE);self.model.eval();self.labels={int(k):v.lower() for k,v in self.model.config.id2label.items()};self.index={v:k for k,v in self.labels.items()};self.fields=list(B);self.hypotheses=[B[f][0].capitalize()+'.' for f in self.fields]
  self.max_hyp=max(len(self.tok(h,add_special_tokens=False)['input_ids']) for h in self.hypotheses);self.budget=512-self.max_hyp-6
 def chunks(self,text):
  ids=self.tok(text,add_special_tokens=False)['input_ids']
  if len(ids)<=self.budget:return [text]
  lines=text.splitlines();chunks=[];current=''
  for line in lines:
   candidate=(current+'\n'+line).strip()
   if len(self.tok(candidate,add_special_tokens=False)['input_ids'])<=self.budget:current=candidate;continue
   if current:chunks.append(current);current=''
   line_ids=self.tok(line,add_special_tokens=False)['input_ids']
   if len(line_ids)>self.budget:
    chunks.extend(self.tok.decode(line_ids[i:i+self.budget]) for i in range(0,len(line_ids),self.budget-32))
   else:current=line
  if current:chunks.append(current)
  return chunks
 def predict(self,text,batch_size=24):
  chunks=self.chunks(text);pairs=[(chunk,h) for chunk in chunks for h in self.hypotheses];probs=[]
  for i in range(0,len(pairs),batch_size):
   rs=pairs[i:i+batch_size];x=self.tok([p[0] for p in rs],[p[1] for p in rs],padding=True,truncation=False,return_tensors='pt');assert x['input_ids'].shape[1]<=512
   with torch.inference_mode():probs.extend(self.model(**x.to(DEVICE)).logits.softmax(-1).cpu().tolist())
  result={};evidence={}
  for j,f in enumerate(self.fields):
   ps=[probs[i*len(self.fields)+j] for i in range(len(chunks))]
   if len(ps)==1:
    label=self.labels[max(range(3),key=lambda k:ps[0][k])];value={'entailment':True,'contradiction':False,'neutral':None}[label]
   else:
    # Neutral in an unrelated chunk does not erase evidence in another chunk.
    ent=max(p[self.index['entailment']] for p in ps);con=max(p[self.index['contradiction']] for p in ps)
    value=True if ent>.5 and con<=.5 else False if con>.5 and ent<=.5 else None
   result[f]=value;evidence[f]=[{self.labels[k]:p[k] for k in range(3)} for p in ps]
  return result,evidence,len(chunks)
 def predict_many(self,texts,batch_size=64):
  chunks=[self.chunks(text) for text in texts];pairs=[];keys=[]
  for ti,cs in enumerate(chunks):
   for ci,chunk in enumerate(cs):
    for fi,h in enumerate(self.hypotheses):pairs.append((chunk,h));keys.append((ti,ci,fi))
  enc=self.tok([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False);order=sorted(range(len(pairs)),key=lambda j:len(enc['input_ids'][j]));assert max(map(len,enc['input_ids']))<=512
  probabilities=[None]*len(pairs)
  for i in range(0,len(order),batch_size):
   ix=order[i:i+batch_size];x=self.tok.pad([{k:enc[k][j] for k in enc} for j in ix],padding=True,return_tensors='pt').to(DEVICE)
   with torch.inference_mode():ps=self.model(**x).logits.float().softmax(-1).cpu().tolist()
   for j,p in zip(ix,ps):probabilities[j]=p
  output=[{f:[None]*len(cs) for f in self.fields} for cs in chunks]
  for (ti,ci,fi),p in zip(keys,probabilities):output[ti][self.fields[fi]][ci]={self.labels[k]:p[k] for k in range(3)}
  return output
if __name__=='__main__':
 from score import evaluate
 torch.set_num_threads(8);fine='--finetuned' in sys.argv;name='finetuned' if fine else 'pretrained';nli=NLI(str(R/'finetuned/best') if fine else None)
 rows=json.loads((R.parent/'varied-training/test.json').read_text());base={r['id']:r['prediction'] for r in json.loads((R.parent/'varied-training/varied_recipe_predictions.json').read_text())};pred=[]
 for r in rows:
  booleans,probs,n=nli.predict(r['text']);p=base[r['id']].copy();p.update(booleans);pred.append({'id':r['id'],'prediction':p,'boolean_probabilities':probs,'chunks':n})
 (R/(name+'_full_predictions.json')).write_text(json.dumps(pred,ensure_ascii=False,indent=2));s=evaluate(pred,rows);(R/(name+'_full_scores.json')).write_text(json.dumps(s,ensure_ascii=False,indent=2));print(name,s['known_value_accuracy'],s['unsupported_values'],s['exact_claims'],flush=True)
