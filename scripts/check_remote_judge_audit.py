"""Run frozen decision-judge audits against a private System One endpoint."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows,json_write,sha256_file,text_hash
from ngila.judge import QUESTIONS,QUESTIONS_V2,parse_response
from ngila.judge_audit import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoint',required=True);p.add_argument('--model',default='xiangxin-2.0.0');p.add_argument('--output',required=True,type=Path);p.add_argument('--controls',nargs='+',required=True,type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    opener=build_opener(ProxyHandler({}))
    manifest={'status':'running','endpoint':a.endpoint,'model':a.model,'threshold':.5,'temperature':'server default; no override','concurrency':1,'input_hashes':{str(x):sha256_file(x) for x in a.controls},'questions':{'v1':QUESTIONS,'v2':QUESTIONS_V2}}
    json_write(a.output/'manifest.json',manifest)
    try:
        for i,path in enumerate(a.controls):
            rows=list(read_rows(path));root=a.output/str(i);root.mkdir();report={}
            for rubric,questions in [('v1',QUESTIONS),('v2',QUESTIONS_V2)]:
                records=[]
                with (root/(rubric+'-decisions.jsonl')).open('x') as f:
                    for row in rows:
                        state=({'user_prompt':row['prompt'],'assistant_response':row['text']} if rubric=='v1' else 'User request:\n'+row['prompt']+'\n\nAssistant response:\n'+row['text'])
                        payload={'model':a.model,'state':state,'questions':questions}
                        request=Request(a.endpoint,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                        started=time.monotonic()
                        with opener.open(request,timeout=45) as response:raw=json.load(response)
                        if raw.get('model')!=a.model:raise ValueError('Unexpected served model')
                        record={'id':row['id'],'output_sha256':text_hash(row['text']),'raw_response':raw,'decisions':parse_response(raw,.5),'seconds':time.monotonic()-started}
                        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();records.append(record)
                report[rubric]=audit(records,rows)
                print(i,rubric,{k:v['accuracy'] for k,v in report[rubric]['fields'].items()},flush=True)
            json_write(root/'report.json',{'input':str(path),'input_sha256':sha256_file(path),'rubrics':report})
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__);raise
    finally:json_write(a.output/'manifest.json',manifest)


if __name__=='__main__':main()
