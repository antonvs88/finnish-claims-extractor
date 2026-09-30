import os,json,time,random,re,math,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
R=Path(__file__).resolve().parent;CAT=json.loads((R.parent/'input_catalog.json').read_text());FIELDS=list(CAT)
CLOSED=[f for f in FIELDS if CAT[f]['type'] in ['boolean','enum']];SPAN=[f for f in FIELDS if f not in CLOSED]
VALUES={f:([None,False,True] if CAT[f]['type']=='boolean' else [None]+list(CAT[f]['values'])) for f in CLOSED}
MODEL=str(R.parent.parent/'models/modernbert-base');REV=None
torch.set_num_threads(8);torch.manual_seed(431);random.seed(431)
TOK=AutoTokenizer.from_pretrained(MODEL,revision=REV);DEVICE='mps'
class Extractor(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=AutoModel.from_pretrained(MODEL,revision=REV,attn_implementation='sdpa',reference_compile=False)
  d=self.encoder.config.hidden_size
  self.queries=nn.Parameter(torch.randn(len(CLOSED),d)*.02)
  self.heads=nn.ModuleList([nn.Linear(d,len(VALUES[f])) for f in CLOSED]);self.span=nn.Linear(d,len(SPAN)*2)
 def forward(self,x):
  h=self.encoder(**x).last_hidden_state
  att=torch.einsum('bld,fd->bfl',h,self.queries)/math.sqrt(h.shape[-1]);att=att.masked_fill(~x['attention_mask'].bool()[:,None,:],-1e4)
  pooled=torch.einsum('bfl,bld->bfd',att.softmax(-1),h)
  cls=[head(pooled[:,i]) for i,head in enumerate(self.heads)]
  sp=self.span(h).permute(0,2,1).masked_fill(~x['attention_mask'].bool()[:,None,:],-1e4)
  return cls,sp

def prepare(rows):
 out=[]
 for r in rows:
  t=TOK(r['text'],return_offsets_mapping=True,truncation=False)
  assert len(t['input_ids'])<=4096,'No silent truncation'
  y=[next(i for i,v in enumerate(VALUES[f]) if type(v)==type(r['expected'][f]) and v==r['expected'][f]) for f in CLOSED]
  z=[]
  for f in SPAN:
   if r['expected'][f] is None:z += [0,0];continue
   a,b=r['spans'][f];ix=[i for i,(s,e) in enumerate(t['offset_mapping']) if e>s and e>a and s<b]
   assert ix and t['offset_mapping'][ix[0]][0]<=a and t['offset_mapping'][ix[-1]][1]>=b
   z += [ix[0],ix[-1]]
  out.append({'ids':t['input_ids'],'y':y,'z':z,'id':r['id']})
 return out

def batch(rs):
 n=max(len(r['ids']) for r in rs);ids=[r['ids']+[TOK.pad_token_id]*(n-len(r['ids'])) for r in rs];mask=[[1]*len(r['ids'])+[0]*(n-len(r['ids'])) for r in rs]
 return ({'input_ids':torch.tensor(ids,device=DEVICE),'attention_mask':torch.tensor(mask,device=DEVICE)},torch.tensor([r['y'] for r in rs],device=DEVICE),torch.tensor([r['z'] for r in rs],device=DEVICE))
def lossfn(c,s,y,z):return torch.stack([F.cross_entropy(v,y[:,i]) for i,v in enumerate(c)]).mean()+F.cross_entropy(s.reshape(-1,s.shape[-1]),z.reshape(-1))
def normalize(s,typ):
 if typ=='date':
  import datetime
  d,m,y=map(int,s.split('.'));return datetime.date(y,m,d).isoformat()
 return float(s.replace(' ','').replace('\u00a0','').replace(',','.'))
MONEY=re.compile(r'(?<![\w.,])(?P<n>\d{1,3}(?:[ \u00a0]\d{3})+(?:,\d{1,2})?|\d+(?:[,.]\d{1,2})?)\s*(?:€|EUR\b|euro\w*)',re.I)
DATE=re.compile(r'(?<!\d)\d{1,2}\.\d{1,2}\.\d{4}(?!\d)')
def predict(model,texts):
 enc=TOK(texts,padding=True,return_tensors='pt',return_offsets_mapping=True,truncation=False);offsets=enc.pop('offset_mapping').tolist()
 x={k:v.to(DEVICE) for k,v in enc.items()}
 with torch.inference_mode():c,s=model(x)
 choices=torch.stack([v.argmax(-1) for v in c],1).cpu().tolist();s=s.cpu();out=[]
 for bi,text in enumerate(texts):
  p={f:VALUES[f][choices[bi][i]] for i,f in enumerate(CLOSED)}
  for j,f in enumerate(SPAN):
   typ=CAT[f]['type'];best=float(s[bi,2*j,0]+s[bi,2*j+1,0]);value=None
   for m in (DATE if typ=='date' else MONEY).finditer(text):
    a,b=m.span() if typ=='date' else m.span('n');ix=[i for i,(u,v) in enumerate(offsets[bi]) if v>u and v>a and u<b]
    if not ix:continue
    sc=float(s[bi,2*j,ix[0]]+s[bi,2*j+1,ix[-1]])
    if sc>best:
     try:value=normalize(text[a:b],typ);best=sc
     except ValueError:pass
   p[f]=value
  out.append(p)
 return out

def main():
 # ONLY train/dev files are opened until training and checkpoint selection finish.
 train=prepare(json.loads((R/'train.json').read_text()));dev=prepare(json.loads((R/'dev.json').read_text()))
 model=Extractor().to(DEVICE)
 for p in model.encoder.parameters():p.requires_grad=False
 for layer in model.encoder.layers[-4:]:
  for p in layer.parameters():p.requires_grad=True
 for p in model.encoder.final_norm.parameters():p.requires_grad=True
 # Initialize attention queries from field definitions, never test examples.
 model.eval()
 with torch.inference_mode():
  x=TOK([CAT[f].get('finnish') or CAT[f]['label'] for f in CLOSED],padding=True,return_tensors='pt').to(DEVICE)
  h=model.encoder(**x).last_hidden_state;v=(h*x['attention_mask'].unsqueeze(-1)).sum(1)/x['attention_mask'].sum(1,keepdim=True)
 model.queries.data.copy_(v)
 heads=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
 enc=[p for p in model.encoder.parameters() if p.requires_grad]
 opt=torch.optim.AdamW([{'params':enc,'lr':3e-5},{'params':heads,'lr':1e-3}],weight_decay=.01)
 meta={'model':MODEL,'revision':REV,'train_count':len(train),'dev_count':len(dev),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'epochs':8,'batch_size':8,'encoder_layers_unfrozen':4,'seed':431,'selection':'lowest dev loss','max_train_tokens':max(len(r['ids']) for r in train),'history':[]}
 (R/'training_plan.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
 best=float('inf');start=time.perf_counter()
 for epoch in range(8):
  model.train();random.shuffle(train);losses=[]
  for i in range(0,len(train),8):
   x,y,z=batch(train[i:i+8]);opt.zero_grad(set_to_none=True);c,s=model(x);loss=lossfn(c,s,y,z);loss.backward();torch.nn.utils.clip_grad_norm_(list(heads)+enc,1.0);opt.step();losses.append(float(loss.detach()))
   if i%128==0:print('epoch',epoch+1,'rows',i,'loss',losses[-1],flush=True)
  model.eval();vl=[]
  with torch.inference_mode():
   for i in range(0,len(dev),8):
    x,y,z=batch(dev[i:i+8]);c,s=model(x);vl.append(float(lossfn(c,s,y,z)))
  row={'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'dev_loss':sum(vl)/len(vl),'elapsed':time.perf_counter()-start};meta['history'].append(row);print(row,flush=True)
  if row['dev_loss']<best:
   best=row['dev_loss'];torch.save(model.state_dict(),R/'best.pt');meta['best_epoch']=epoch+1
  (R/'training_result.json').write_text(json.dumps(meta,indent=2))
 model.load_state_dict(torch.load(R/'best.pt',map_location=DEVICE,weights_only=True));model.eval()
 for split in ['test','challenge']:
  rows=json.loads((R/(split+'.json')).read_text());predict(model,['Asiakas hakee korvausta 100 euroa.']);preds=[]
  for r in rows:
   torch.mps.synchronize();st=time.perf_counter();p=predict(model,[r['text']])[0];torch.mps.synchronize();sec=time.perf_counter()-st;preds.append({'id':r['id'],'prediction':p,'seconds':sec})
  (R/('modernbert_'+split+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in preds))
 print('DONE',flush=True)
if __name__=='__main__':main()
