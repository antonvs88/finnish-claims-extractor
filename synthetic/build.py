"""Build the synthetic boolean question-answer set: python synthetic/build.py

dev: question wordings 0-2, wrappers 0-1, answer wordings 0-1 (432 cases).
holdout: question wording 3, answer wording 2, wrappers 1-2 - none of the three seen in dev (144 cases).
smoke: the first case of each field and class in dev (72), the quick check for a pull request.
Deterministic, no randomness.
"""
import json
from pathlib import Path
from phrasings import QUESTIONS,ANSWERS,WRAPPERS
HERE=Path(__file__).resolve().parent
GOLD={'true':True,'false':False,'null':None}
def cases():
    out=[]
    for field,qs in QUESTIONS.items():
        assert len(qs)==4,field
        for cls in ('true','false','null'):
            combos=[('dev',p,k%2,k%2) for p in range(3) for k in range(2)]+[('holdout',3,2,2),('holdout',3,1,2)]
            for split,p,w,a in combos:
                text=WRAPPERS[w].format(q=qs[p],a=ANSWERS[cls][a])
                out.append({'id':f'{field}|{cls}|{split}|p{p}w{w}a{a}','split':split,'field':field,'gold':GOLD[cls],'text':text})
    return out
def main():
    rows=cases();(HERE/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    seen=set();smoke=[]
    for r in rows:
        key=(r['field'],r['gold'])
        if r['split']=='dev' and key not in seen:seen.add(key);smoke.append(r)
    for name,sel in (('dev',[r for r in rows if r['split']=='dev']),('holdout',[r for r in rows if r['split']=='holdout']),('smoke',smoke)):
        (HERE/f'input-{name}.json').write_text(json.dumps([{'id':r['id'],'text':r['text']} for r in sel],ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(name,len(sel))
if __name__=='__main__':main()
