"""Select a reproducible balanced judge audit from locally authorized WildGuard data."""
import argparse
import hashlib
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows, jsonl_write, json_write, sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/processed/v0.2/test.jsonl');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    rows=[r for r in read_rows(a.input) if r['source']=='wildguard'];selected=[]
    for label in ('harmful','unharmful'):
        eligible=[r for r in rows if r['source_labels']['response_harm_label']==label and isinstance(r['reference_response'],str) and r['reference_response'].strip()]
        eligible.sort(key=lambda r:hashlib.sha256(('intern-judge-audit-v1:'+r['id']).encode()).hexdigest())
        seen=set();chosen=[]
        for r in eligible:
            if r['prompt_key'] in seen:continue
            seen.add(r['prompt_key']);chosen.append(r)
            if len(chosen)==100:break
        if len(chosen)!=100:raise ValueError('Insufficient examples')
        for r in chosen:
            refusal=r['source_labels']['response_refusal_label']
            selected.append({'id':r['id'],'prompt':r['prompt'],'text':r['reference_response'],'split':'judge_audit','origin':'wildguardtest','category':label,'language':r['language'],'expected_refusal':{'refusal':True,'compliance':False}.get(refusal),'expected_fulfilled':None,'expected_unsafe':label=='harmful','source_labels':r['source_labels'],'source_revision':r['source_revision'],'label_author':'WildGuard dataset annotators','source_url':r['source_url'],'license':r['license']})
    jsonl_write(root/'controls.jsonl',selected)
    json_write(root/'selection.json',{'n':len(selected),'seed':'intern-judge-audit-v1','source_revision':sorted({r['source_revision'] for r in selected}),'source_split':'wildguardtest','stratification':'100 harmful and 100 unharmful responses; prompt dedup within stratum','refusal_labels':dict(Counter(str(r['expected_refusal']) for r in selected)),'controls_sha256':sha256_file(root/'controls.jsonl'),'redistribution':'Gated source text remains local'})


if __name__=='__main__':main()
