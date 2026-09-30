"""Joint numeric-candidate roles + existing enum heads, one encoder forward pass."""
import importlib.util,json,sys,math
from pathlib import Path
import torch
from torch import nn
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'training'))
spec=importlib.util.spec_from_file_location('claims_base',R.parent/'training/train.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
CLASSES=[None]+t.SPAN
class Joint(t.Extractor):
 def __init__(self):
  super().__init__();self.role_head=nn.Linear(self.encoder.config.hidden_size,len(CLASSES))
 def forward(self,x):
  h=self.encoder(**x).last_hidden_state;att=torch.einsum('bld,fd->bfl',h,self.queries)/math.sqrt(h.shape[-1]);att=att.masked_fill(~x['attention_mask'].bool()[:,None,:],-1e4);pooled=torch.einsum('bfl,bld->bfd',att.softmax(-1),h);cls=[head(pooled[:,i]) for i,head in enumerate(self.heads)]
  return cls,self.role_head(h)
def candidates(text,offsets):
 out=[]
 for typ,pat in [('number',t.MONEY),('date',t.DATE)]:
  for m in pat.finditer(text):
   a,b=m.span('n') if typ=='number' else m.span();ix=[i for i,(u,v) in enumerate(offsets) if v>u and v>a and u<b]
   if ix:out.append({'span':[a,b],'token':ix[0],'type':typ,'value':t.normalize(text[a:b],typ)})
 return out
@torch.inference_mode()
def predict(model,texts,threshold=.5,enum_null_factor=1.0):
 x=t.TOK(texts,padding=True,return_tensors='pt',return_offsets_mapping=True,truncation=False);offsets=x.pop('offset_mapping').tolist();cs,role=model(x.to(next(model.parameters()).device));choices=torch.stack([(v + torch.tensor([math.log(enum_null_factor)]+[0.]*(v.shape[-1]-1),device=v.device) if t.CAT[t.CLOSED[j]]['type']=='enum' else v).argmax(-1) for j,v in enumerate(cs)],1).cpu().tolist();role=role.float().softmax(-1).cpu();out=[]
 for bi,text in enumerate(texts):
  p={f:t.VALUES[f][choices[bi][j]] for j,f in enumerate(t.CLOSED)};p.update(dict.fromkeys(t.SPAN));best=dict.fromkeys(t.SPAN,threshold)
  for c in candidates(text,offsets[bi]):
   scores=role[bi,c['token']];k=int(scores.argmax());f=CLASSES[k]
   if f is not None and t.CAT[f]['type']==c['type'] and float(scores[k])>best[f]:best[f]=float(scores[k]);p[f]=c['value']
  out.append(p)
 return out
