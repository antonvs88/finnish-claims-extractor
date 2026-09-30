"""Experimental closed-code aggregation; no data-dependent runtime rules."""
def decode(raw,factor,margin,mode='global_max'):
 out={}
 for field in sorted({k.split('::')[0] for k in raw}):
  keys=[k for k in raw if k.split('::')[0]==field];counts={len(raw[k]) for k in keys};assert len(counts)==1
  scores=[[(raw[k][i]['entailment']/(raw[k][i]['entailment']+raw[k][i]['contradiction']+factor*raw[k][i]['neutral']),k.split('::')[1]) for k in keys] for i in range(next(iter(counts)))]
  if mode=='global_max':
   ranked=sorted([(max(row[j][0] for row in scores),keys[j].split('::')[1]) for j in range(len(keys))],reverse=True);out[field]=ranked[0][1] if ranked[0][0]>.5 and ranked[0][0]-ranked[1][0]>=margin else None
  else:
   votes=[]
   for row in scores:
    ranked=sorted(row,reverse=True);gap=ranked[0][0]-ranked[1][0]
    if ranked[0][0]>.5 and gap>=margin:votes.append((gap,ranked[0][0],ranked[0][1]))
   if mode=='consistent_chunks':out[field]=votes[0][2] if votes and len({v[2] for v in votes})==1 else None
   elif mode=='strongest_chunk':out[field]=max(votes)[2] if votes else None
   else:raise ValueError(mode)
 return out
