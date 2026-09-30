"""Score extractor output on the synthetic set, in aggregate.

python synthetic/score.py PREDICTIONS.json --split dev|holdout|smoke [--baseline synthetic/baseline-SPLIT.json] [--write-baseline PATH]

PREDICTIONS.json is the output of run.py on synthetic/input-SPLIT.json. Prints per-field counts
and totals only, never a case text. With --baseline the exit code is 1 when the run is worse than the
accepted one: a field with fewer correct answers, more wrong answers (true/false swapped, or an
answer where the truth is unknown), or more spurious answers on other boolean fields.
"""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def name(v):return {True:'true',False:'false',None:'null'}[v]
def score(preds,split):
    cases={c['id']:c for c in json.loads((HERE/'cases.json').read_text(encoding='utf-8')) if c['split']==split or split=='smoke'}
    if split=='smoke':cases={i:c for i,c in cases.items() if i in {r['id'] for r in json.loads((HERE/'input-smoke.json').read_text(encoding='utf-8'))}}
    booleans={c['field'] for c in cases.values()}
    got={r['id']:r['prediction'] for r in preds}
    assert set(got)==set(cases),'Predictions do not match the split'
    fields={};spurious=0
    for i,c in cases.items():
        f=c['field'];p=got[i][f];g=c['gold'];d=fields.setdefault(f,{'cases':0,'correct':0,'wrong':0,'missed':0,'confusion':{}})
        d['cases']+=1;key=f'{name(g)}>{name(p)}';d['confusion'][key]=d['confusion'].get(key,0)+1
        if p==g:d['correct']+=1
        elif p is None:d['missed']+=1
        else:d['wrong']+=1
        spurious+=sum(1 for k in booleans if k!=f and got[i].get(k) is not None)
    total={k:sum(d[k] for d in fields.values()) for k in ('cases','correct','wrong','missed')}
    return {'split':split,'total':total,'spurious':spurious,'fields':dict(sorted(fields.items()))}
def regressions(new,base):
    out=[]
    for f,d in new['fields'].items():
        b=base['fields'][f]
        if d['correct']<b['correct']:out.append(f'{f}: correct {b["correct"]} -> {d["correct"]}')
        if d['wrong']>b['wrong']:out.append(f'{f}: wrong {b["wrong"]} -> {d["wrong"]}')
    if new['spurious']>base['spurious']:out.append(f'spurious {base["spurious"]} -> {new["spurious"]}')
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('predictions');ap.add_argument('--split',required=True,choices=['dev','holdout','smoke'])
    ap.add_argument('--baseline');ap.add_argument('--write-baseline');a=ap.parse_args()
    result=score(json.loads(Path(a.predictions).read_text(encoding='utf-8')),a.split)
    t=result['total'];print(f"{a.split}: {t['correct']}/{t['cases']} correct, {t['wrong']} wrong, {t['missed']} missed (null instead of an answer), spurious {result['spurious']}")
    for f,d in result['fields'].items():
        if d['correct']!=d['cases']:print(f"  {f}: {d['correct']}/{d['cases']} wrong {d['wrong']} missed {d['missed']}")
    if a.write_baseline:Path(a.write_baseline).write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8')
    if a.baseline:
        bad=regressions(result,json.loads(Path(a.baseline).read_text(encoding='utf-8')))
        print('REGRESSION' if bad else 'no regression against the baseline',*bad,sep='\n  ')
        sys.exit(1 if bad else 0)
main()
