"""Run a supervised S1 judge experiment, separate from ngila direction fitting."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import sha256_file, json_write, read_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    root, data, runtime = map(Path, (args.out, args.data, args.runtime))
    if root.exists():
        raise FileExistsError(root)
    manifest = json.loads((data / 'manifest.json').read_text())
    keys = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        path = data / (split + '.jsonl')
        if sha256_file(path) != manifest['files'][split]['sha256']:
            raise ValueError('Dataset changed: ' + split)
        keys[split] = {r['prompt_key'] for r in read_rows(path)}
        if any(keys[split] & keys[s] for s in keys if s != split):
            raise ValueError('Cross-split prompt leakage')
    root.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(runtime.resolve()), HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    common = ['--general-train', str(data / 'train.jsonl'), '--general-val', str(data / 'validation.jsonl'),
              '--model', args.model, '--head', 'linear', '--anchor', 'text', '--attention', 'causal',
              '--bf16-weights', '--max-state-tokens', str(manifest['max_state_tokens']),
              '--batch-tokens', '8192', '--seed', '20261012']
    commands = {
        'baseline': ['--out', str(root / 'baseline'), '--eval-only'],
        'train': ['--out', str(root / 'fit'), '--epochs', '1', '--lr', '1e-5',
                  '--head-lr', '1e-4', '--warmup', '10', '--optimizer', 'adamw_bf16sr',
                  '--freeze-regex', r'^(?!backbone\.layers\.(28|29|30|31)\.|head\.)']}
    record = {'status': 'prepared', 'purpose': 'neutral_response_classification',
              'model': args.model, 'data_manifest_sha256': sha256_file(data / 'manifest.json'),
              'runtime': str(runtime.resolve()), 'python': sys.executable,
              'runtime_hashes': {str(p.relative_to(runtime)): sha256_file(p)
                                 for p in sorted((runtime / 'systemone').rglob('*.py'))},
              'source_metadata': json.loads((Path(args.model) / 'systemone.json').read_text()),
              'source_weights_sha256': sha256_file(Path(args.model) / 'model.safetensors'),
              'upstream_evaluation_contamination': 'Known upstream evaluation-set training; new holdouts not certified unseen',
              'temperature': 'raw logits at T=1 for both models; post-fit calibration pending',
              'tasks': ['refusal', 'response_unsafe'], 'request_fulfilled': 'not trained',
              'started_at_unix': time.time(), 'commands': {}, 'gpu': args.gpu}
    json_write(root / 'manifest.json', record)
    try:
        for phase, extra in commands.items():
            cmd = [sys.executable, str(Path(__file__).with_name('gpu_run.py')), '--gpu', str(args.gpu),
                   '--', sys.executable, '-m', 'systemone.train.train', *common, *extra]
            record.update(status='running', phase=phase)
            record['commands'][phase] = cmd
            json_write(root / 'manifest.json', record)
            print(f'{phase}: {root / (phase + ".log")}', flush=True)
            with (root / (phase + '.log')).open('x') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        history = json.loads((root / 'fit/history.json').read_text())
        if not history or not all(math.isfinite(h['mean_nll']) for h in history):
            raise ValueError('Invalid validation history')
        if not (root / 'fit/best/readout_head.pt').is_file():
            raise ValueError('No saved best checkpoint')
        record.update(status='fit_complete_not_accepted', phase='validation_complete', finished_at_unix=time.time())
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), finished_at_unix=time.time())
        raise
    finally:
        json_write(root / 'manifest.json', record)


if __name__ == '__main__':
    main()
