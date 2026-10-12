"""Fit reversible steering directions from reviewed training-split behavior."""
import json
import subprocess
from pathlib import Path

from .evaluation import evaluate
from .io import json_write, jsonl_write, read_rows, sha256_file


def discovery(outputs, annotations, destination):
    evaluate(outputs,annotations)  # Validate binding and reviewer metadata first.
    labels = {a['id']:a for a in read_rows(annotations)}
    result = []
    for row in read_rows(outputs):
        if row['split'] != 'train':
            raise ValueError('Only train split generations may produce discovery data')
        if row.get('truncated') is not False or row.get('ended_with_eos') is not True:
            continue
        a = labels.get(row['id'],{})
        behavior = a.get('behavior')
        if behavior not in ('full_refusal','deflection','answer'):
            continue
        group = 'refusal' if behavior in ('full_refusal','deflection') else 'answer'
        result.append({**row,'split':'discovery','group':group,
                       'behavior_annotation_sha256':sha256_file(annotations)})
    if any(sum(r['group']==group for r in result)<2 for group in ('refusal','answer')):
        raise ValueError('Need at least two reviewed completed examples of each group')
    if Path(destination).exists():
        raise FileExistsError(destination)
    jsonl_write(destination,result)
    return {'records':len(result),'groups':{g:sum(r['group']==g for r in result) for g in ('refusal','answer')},
            'limitation':'Behavioral contrast may confound topic/style. Validate with matched controls.'}


def train(model, prompts, output, layer, python, device='cuda'):
    rows = list(read_rows(prompts))
    if not rows or any(r['split']!='discovery' for r in rows):
        raise ValueError('Training requires a discovery-only file')
    if any(sum(r.get('group')==g for r in rows)<2 for g in ('refusal','answer')):
        raise ValueError('Need refusal and answer groups (at least two each)')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    commands = [
        [python,str(repo/'scripts/experiment.py'),'--model',str(model),'--prompts',str(prompts),
         '--output',str(output/'collect'),'--mode','collect','--device',device],
        [python,str(repo/'scripts/direction.py'),'--run',str(output/'collect'),'--positive','refusal',
         '--negative','answer','--layer',str(layer),'--output',str(output/'direction.pt')],
    ]
    record = {'kind':'activation_steering','status':'running','base_model':str(model),
              'layer':layer,'prompts_sha256':sha256_file(prompts),'commands':commands,
              'limitation':'Not a standalone LLM or permanent weight edit; evaluation pending'}
    try:
        for i,cmd in enumerate(commands):
            with (output/f'stage-{i}.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        record.update(status='fitted_unvalidated',direction_sha256=sha256_file(output/'direction.pt'))
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__)
        raise
    finally:
        json_write(output/'artifact.json',record)
    return record
