"""Entailment hypotheses for all boolean and closed-code fields; no gold-dependent inference."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P/'nli-baseline'))
from inference import NLI
from calibrate import classify
from build_data import B,CAT
BF=list(B);EF=[f for f,c in CAT.items() if c['type']=='enum']
PREFIX={'damageCause':'Tämän vahingon tapahtumalaji on','korvaustapa':'Fennia on valinnut tämän vahingon korvaustavaksi','kustannuksenAlvKasittely':'Käsittelijän vahvistama korjaus- tai hankintakulujen ALV-kohtelu on seuraava:','lunastusarvonPeruste':'Käsittelijän vahvistama lunastusarvon peruste on','muunVeronTaiMaksunKasittely':'Käsittelijän vahvistama muun ajoneuvoveron tai maksun käsittely on seuraava:','lisaomavastuunAjoneuvoluokka':'Ajoneuvorekisteristä tarkistettu ajoneuvoluokka on'}
KEYS=list(B);HYP={f:B[f][0].capitalize()+'.' for f in B};MAP={}
for f in EF:
 for code,desc in CAT[f]['values'].items():
  key=f+'::'+code;KEYS.append(key);HYP[key]=PREFIX[f]+' '+desc.lower()+'.';MAP[key]=(f,code)
class SchemaNLI(NLI):
 def __init__(self,path,mode='all'):
  super().__init__(str(path));self.fields=KEYS if mode=='all' else list(MAP) if mode=='enum' else BF;self.hypotheses=[HYP[k] for k in self.fields];self.max_hyp=max(len(self.tok(h,add_special_tokens=False)['input_ids']) for h in self.hypotheses);self.budget=512-self.max_hyp-6

def decode_bool(probs,factor):return classify({f:probs[f] for f in BF},factor)
def decode_enum(probs,factor=1,margin=0):
 out={}
 for f in EF:
  ranked=[]
  for code in CAT[f]['values']:
   ps=probs[f+'::'+code];scores=[p['entailment']/(p['entailment']+p['contradiction']+factor*p['neutral']) for p in ps];ranked.append((max(scores),code))
  ranked.sort(reverse=True);best,code=ranked[0];second=ranked[1][0];out[f]=code if best>.5 and best-second>=margin else None
 return out

def metrics(rows,ps,fields):
 from score import correct
 known=sum(r['expected'][f] is not None for r in rows for f in fields);ans=sum(p[f] is not None for p in ps for f in fields);tp=sum(r['expected'][f] is not None and correct(p[f],r['expected'][f]) for r,p in zip(rows,ps) for f in fields);return {'correct':tp,'known':known,'answered':ans,'recall':tp/known if known else 0,'precision':tp/ans if ans else 0,'f1':2*tp/(known+ans) if known+ans else 0}
def calibrate(rows,probs):
 groups={name:[i for i,r in enumerate(rows) if r['dev_group']==name] for name in sorted({r['dev_group'] for r in rows})}
 def group_metric(ps,fields):
  out={name:metrics([rows[i] for i in ix],[ps[i] for i in ix],fields) for name,ix in groups.items()};return sum(x['f1'] for x in out.values())/len(out),out
 bg=[];eg=[]
 for factor in [.125,.25,.5,1,2,4,8,16,32,64,128,256]:
  ps=[decode_bool(p,factor) for p in probs];metric,gs=group_metric(ps,BF);bg.append({'factor':factor,'metric':metric,'groups':gs,'overall':metrics(rows,ps,BF)})
 for factor in [.25,.5,1,2,4,8,16,32,64]:
  for margin in [0,.1]:
   ps=[decode_enum(p,factor,margin) for p in probs];metric,gs=group_metric(ps,EF);eg.append({'factor':factor,'margin':margin,'metric':metric,'groups':gs,'overall':metrics(rows,ps,EF)})
 b=max(bg,key=lambda x:(x['metric'],x['overall']['precision']));e=max(eg,key=lambda x:(x['metric'],x['overall']['precision']));return {'boolean':b,'enum':e,'balanced_metric':(b['metric']+e['metric'])/2,'boolean_grid':bg,'enum_grid':eg}
