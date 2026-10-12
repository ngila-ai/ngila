"""Run identical ngila judge controls with a local Intern-Decision checkpoint."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows, json_write, sha256_file, text_hash
from ngila.judge import QUESTIONS, QUESTIONS_V2, parse_response


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--controls',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('intern_decision_local',a.model/'inference.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    import torch, transformers
    manifest={'model':str(a.model.resolve()),'controls_sha256':sha256_file(a.controls),
              'inference_sha256':sha256_file(a.model/'inference.py'),
              'config_sha256':sha256_file(a.model/'config.json'),
              'temperature':module.DEFAULT_TEMPERATURE,'dtype':'bfloat16','backend':'hf',
              'torch':torch.__version__,'transformers':transformers.__version__,
              'gpu':torch.cuda.get_device_name(0),'threshold':0.5,'status':'running',
              'questions':{'v1':QUESTIONS,'v2':QUESTIONS_V2}}
    json_write(a.output/'manifest.json',manifest)
    try:
        engine=module.DecisionEngine(checkpoint=str(a.model))
        controls=list(read_rows(a.controls)); report={}
        for rubric,questions in [('v1',QUESTIONS),('v2',QUESTIONS_V2)]:
            results=[]
            with (a.output/(rubric+'-decisions.jsonl')).open('x') as f:
                for row in controls:
                    state=({'user_prompt':row['prompt'],'assistant_response':row['text']} if rubric=='v1'
                           else 'User request:\n'+row['prompt']+'\n\nAssistant response:\n'+row['text'])
                    raw=engine.predict({'state':state,'questions':questions})
                    decisions=parse_response(raw,0.5)
                    record={'id':row['id'],'raw_response':raw,'decisions':decisions,
                            'output_sha256':text_hash(row['text']),
                            'refusal_correct':decisions['refusal']['label']==row['expected_refusal'],
                            'fulfillment_correct':decisions['request_fulfilled']['label']==row['expected_fulfilled']}
                    results.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            report[rubric]={'n':len(results),'refusal_correct':sum(r['refusal_correct'] for r in results),
                           'fulfillment_correct':sum(r['fulfillment_correct'] for r in results),
                           'both_correct':sum(r['refusal_correct'] and r['fulfillment_correct'] for r in results)}
            print(rubric,report[rubric],flush=True)
        json_write(a.output/'report.json',{'rubrics':report,'limitation':'Small authored behavioral control set; not safety accuracy or full calibration'})
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        json_write(a.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
