"""Command-array pipeline; append provenance to a durable experiment ledger."""
import datetime
import json
import subprocess
from pathlib import Path

from .io import json_write, sha256_file


def run(config_path, output, ledger='docs/experiments.jsonl'):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    steps = config['steps']
    for step in steps:
        if not isinstance(step['argv'],list) or not step['argv'] or not all(isinstance(x,str) for x in step['argv']):
            raise ValueError('Each step must use an argv string list, never a shell command')
    output.mkdir(parents=True)
    record = {'id':config['id'],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'config_sha256':sha256_file(config_path),'status':'running','steps':[]}
    record['code_sha256'] = {str(p):sha256_file(p) for directory in ('ngila','scripts')
                             for p in sorted(Path(directory).glob('*.py'))}
    json_write(output/'config.json',config)
    try:
        for i,step in enumerate(steps):
            log = output/f'{i:02d}.log'
            entry = {'name':step['name'],'argv':step['argv'],
                     'inputs':{p:sha256_file(p) for p in step.get('inputs',[])}}
            with log.open('w') as f:
                result = subprocess.run(step['argv'],stdout=f,stderr=subprocess.STDOUT,check=False)
            entry.update(returncode=result.returncode, log_sha256=sha256_file(log))
            record['steps'].append(entry)
            if result.returncode:
                raise RuntimeError(f'Step {step["name"]} failed; see {log}')
            entry['outputs'] = {p:sha256_file(p) for p in step.get('outputs',[])}
            json_write(output/'run.json',record)
        record['status'] = 'complete'
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__)
        raise
    finally:
        record['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json_write(output/'run.json',record)
        Path(ledger).parent.mkdir(parents=True,exist_ok=True)
        with Path(ledger).open('a') as f:
            f.write(json.dumps({**record,'run_path':str(output)},ensure_ascii=False)+'\n')
    return record


def render_report(reports, output):
    lines = ['# Generated experiment results', '',
             'Rates are conditional on reviewed, completed outputs. No single anti-alignment score.', '',
             '| Run | Attempted | Completed | Reviewed | Refusal/deflection | Factual correct (reviewed subset) |',
             '|---|---:|---:|---:|---|---|']
    def rate(value):
        return 'pending' if not value['n'] else f'{value["numerator"]}/{value["n"]} ({value["value"]:.1%})'
    for p in reports:
        report = json.loads(Path(p).read_text()); r = report['overall']
        lines.append(f'| {Path(p).stem} | {r["attempted"]} | {r["completed"]} | {r["annotated_completed"]} | '
                     f'{rate(r["refusal_or_deflection_rate"])} | {rate(r["factual_correct_rate"])} |')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text('\n'.join(lines)+'\n')
    return {'reports':len(reports),'output':str(output)}
