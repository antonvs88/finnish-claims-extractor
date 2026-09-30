import math


class ProbabilityEnsemble:
 def __init__(self,base,new,weight,method):
  assert 0<weight<1 and method in ['arithmetic','geometric']
  self.base=base;self.new=new;self.weight=weight;self.method=method
 def predict_many(self,texts,batch_size=8):
  old=self.base.predict_many(texts,batch_size=batch_size);new=self.new.predict_many(texts,batch_size=batch_size);merged=[]
  for bp,np in zip(old,new):
   assert set(bp)==set(np);item={}
   for f,bchunks in bp.items():
    assert len(bchunks)==len(np[f]),'Ensemble chunk boundaries must match';item[f]=[]
    for b,n in zip(bchunks,np[f]):
     w=self.weight;p={k:((1-w)*b[k]+w*n[k] if self.method=='arithmetic' else math.exp((1-w)*math.log(max(b[k],1e-30))+w*math.log(max(n[k],1e-30)))) for k in b};z=sum(p.values());item[f].append({k:v/z for k,v in p.items()})
   merged.append(item)
  return merged

def enum_values(raw,factor,margin):
 out={}
 for field in {key.split('::')[0] for key in raw}:
  ranked=[]
  for key,ps in raw.items():
   f,code=key.split('::')
   if f!=field:continue
   ranked.append((max(p['entailment']/(p['entailment']+p['contradiction']+factor*p['neutral']) for p in ps),code))
  ranked.sort(reverse=True);out[field]=ranked[0][1] if ranked[0][0]>.5 and ranked[0][0]-ranked[1][0]>=margin else None
 return out
