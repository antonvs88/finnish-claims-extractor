"""Retrieve mentioned country hypotheses; NLI must establish event location, not mere mention."""
import re,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(P/'nli-baseline'))
from inference import NLI,DEVICE
COUNTRIES=[('suom', 'Suomessa',True),('ruots','Ruotsissa',True),('norj','Norjassa',True),('tansk','Tanskassa',True),('islan','Islannissa',True),('saks','Saksassa',False),('viro|viross|viron','Virossa',False),('ransk','Ranskassa',False),('belgi','Belgiassa',False),('puol','Puolassa',False),('ital','Italiassa',False),('espan','Espanjassa',False),('alankom','Alankomaissa',False)]
class CountryNLI(NLI):
 def predict_country(self,texts,batch_size=8):
  pairs=[];keys=[];result=[[] for t in texts]
  for i,text in enumerate(texts):
   candidates=[('Pohjoismaissa',True),('Pohjoismaiden ulkopuolella',False)]+[(loc,val) for pattern,loc,val in COUNTRIES if re.search(pattern,text,re.I)]
   for chunk in self.chunks(text):
    for loc,val in candidates:pairs.append((chunk,f'Vahinko sattui {loc}.'));keys.append((i,loc,val))
  enc=self.tok([x[0] for x in pairs],[x[1] for x in pairs],truncation=False);assert max(map(len,enc['input_ids']))<=512;order=sorted(range(len(pairs)),key=lambda i:len(enc['input_ids'][i]))
  for offset in range(0,len(order),batch_size):
   ix=order[offset:offset+batch_size];x=self.tok.pad([{k:v[j] for k,v in enc.items()} for j in ix],padding=True,return_tensors='pt').to(DEVICE)
   with torch.inference_mode():ps=self.model(**x).logits.float().softmax(-1).cpu().tolist()
   for j,p in zip(ix,ps):
    i,loc,val=keys[j];result[i].append({'location':loc,'nordic':val,'probabilities':{self.labels[k]:p[k] for k in range(3)}})
  return result
def decode(records,factor,threshold):
 accepted=set()
 for r in records:
  p=r['probabilities'];s=p['entailment']/(p['entailment']+p['contradiction']+factor*p['neutral'])
  if s>threshold:accepted.add(r['nordic'])
 return next(iter(accepted)) if len(accepted)==1 else None
