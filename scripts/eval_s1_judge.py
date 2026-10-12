"""Save judge logits, calibrate only on calibration split, evaluate frozen test."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import json_write, jsonl_write, read_rows, sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--runtime', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.runtime)
    import torch
    from systemone.train.train import load_tokenizer
    from systemone.train.data import load_general_examples, batches_by_tokens
    from systemone.model.readout import ReadoutModel
    from systemone.train.metrics import QuestionMetrics
    root, data = Path(args.out), Path(args.data)
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / 'manifest.json').read_text())
    tok, anchors, sep = load_tokenizer(args.model, 'text')
    pad = tok.pad_token_id or 0
    model = ReadoutModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda').eval()
    report = {'model': args.model, 'options_order': ['yes', 'no'], 'threshold': 0.5,
              'calibration_method': 'one temperature across both tasks, calibration NLL grid minimization',
              'test_used_for_selection': False, 'source_manifest_sha256': sha256_file(data / 'manifest.json'),
              'prior_contamination': manifest['limitations'], 'splits': {}}
    temperature = None
    for split in ('calibration', 'test'):
        path = data / (split + '.jsonl')
        if sha256_file(path) != manifest['files'][split]['sha256']:
            raise ValueError('Dataset hash mismatch: ' + split)
        examples = load_general_examples(path, tok, anchor_ids=anchors, sep_id=sep,
                    bos_id=tok.bos_token_id, max_state_tokens=manifest['max_state_tokens'])
        if any(e.encoded.truncated_state for e in examples):
            raise ValueError('Truncation is forbidden')
        records = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for group in batches_by_tokens(examples, 8192):
                logits = model.anchor_logits_causal([e.encoded for e in group], pad_id=pad, device='cuda')
                for e, qs in zip(group, logits):
                    values = qs[0].float().cpu().tolist()
                    if not all(math.isfinite(x) for x in values):
                        raise ValueError('Nonfinite logits')
                    records.append({'id': e.key, 'task': e.task, 'label': e.labels[0],
                                    'logits': values, 'truncated': False})
        if (len(records) != manifest['files'][split]['decisions']
                or {r['id'] for r in records} != {e.key for e in examples}):
            raise ValueError('Missing or duplicated predictions')
        if split == 'calibration':
            logits = torch.tensor([r['logits'] for r in records])
            labels = torch.tensor([r['label'] for r in records])
            grid = torch.exp(torch.linspace(math.log(0.2), math.log(5), 161))
            losses = [float(torch.nn.functional.cross_entropy(logits / t, labels)) for t in grid]
            index = min(range(len(losses)), key=losses.__getitem__)
            temperature = float(grid[index])
            report.update(temperature=temperature, temperature_at_boundary=index in (0, len(grid) - 1))
            json_write(root / 'calibration.json', {'temperature': temperature, 'split': split,
                                                  'sha256': sha256_file(path), 'test_used': False})
        metrics = {}
        for name, temp in [('raw_T1', 1.0), ('calibrated', temperature)]:
            groups = defaultdict(QuestionMetrics)
            confusion = defaultdict(lambda: dict(tp=0, tn=0, fp=0, fn=0))
            for r in records:
                probs = torch.softmax(torch.tensor(r['logits']) / temp, -1).tolist()
                r[name + '_yes_probability'] = probs[0]
                groups[r['task']].add(probs, r['label'])
                predicted, actual = probs[0] >= .5, r['label'] == 0
                key = ('tp' if actual else 'fp') if predicted else ('fn' if actual else 'tn')
                confusion[r['task']][key] += 1
            metrics[name] = {task: {**m.finalize().summary(), **confusion[task]} for task, m in groups.items()}
        report['splits'][split] = metrics
        jsonl_write(root / (split + '-predictions.jsonl'), records)
        json_write(root / 'report.json', report)
        print(json.dumps({'split': split, 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
