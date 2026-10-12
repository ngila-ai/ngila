"""Compare local d3 on the same frozen judge audit inputs and rubrics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows,json_write,sha256_file,text_hash
from ngila.judge import QUESTIONS,QUESTIONS_V2,parse_response
from ngila.judge_audit import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--controls',required=True,nargs='+',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.model.resolve()))
    from d3_runtime import D3
    import torch,transformers
    torch.set_num_threads(8)
    manifest={'status':'running','model':str(a.model.resolve()),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'threshold':.5,'temperature':'checkpoint default; no override','question_rubrics':{'v1':QUESTIONS,'v2':QUESTIONS_V2},'input_hashes':{str(p):sha256_file(p) for p in a.controls},'runtime_hashes':{p.name:sha256_file(p) for p in a.model.glob('*.py')}}
    json_write(a.output/'manifest.json',manifest)
    try:
        model=D3.from_pretrained(a.model,device='cuda:0',local_files_only=True,verify='fast')
        manifest['provenance']=model.provenance();json_write(a.output/'manifest.json',manifest)
        for i,path in enumerate(a.controls):
            rows=list(read_rows(path));root=a.output/str(i);root.mkdir();report={}
            for rubric,questions in [('v1',QUESTIONS),('v2',QUESTIONS_V2)]:
                records=[]
                with (root/(rubric+'-decisions.jsonl')).open('x') as f:
                    for row in rows:
                        state=({'user_prompt':row['prompt'],'assistant_response':row['text']} if rubric=='v1' else 'User request:\n'+row['prompt']+'\n\nAssistant response:\n'+row['text'])
                        raw=model.system_one(state=state,questions=questions)
                        decisions=parse_response(raw,.5)
                        r={'id':row['id'],'output_sha256':text_hash(row['text']),'raw_response':raw,'decisions':decisions}
                        records.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                report[rubric]=audit(records,rows)
                print(i,rubric,{k:v['accuracy'] for k,v in report[rubric]['fields'].items()},flush=True)
            json_write(root/'report.json',{'input':str(path),'input_sha256':sha256_file(path),'rubrics':report})
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__);raise
    finally:json_write(a.output/'manifest.json',manifest)


if __name__=='__main__':main()
