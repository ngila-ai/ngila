"""Independently recount paired held-out predictions, without source text."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows, json_write, sha256_file
from ngila.evaluation import proportion


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    roots = {'baseline': Path(args.baseline), 'candidate': Path(args.candidate)}
    reports = {name: json.loads((root / 'report.json').read_text()) for name, root in roots.items()}
    if reports['baseline']['source_manifest_sha256'] != reports['candidate']['source_manifest_sha256']:
        raise ValueError('Different datasets')
    predictions = {}
    result = {'source_manifest_sha256': reports['baseline']['source_manifest_sha256'],
              'models': {}, 'paired_changes': {}, 'accepted_for_default_use': False,
              'limitations': reports['candidate']['prior_contamination'] + [
                  'Fulfillment, Chinese, long inputs and injection robustness not validated',
                  'Parent checkpoint is not certified bit-identical to the remote served S1 export',
                  'Classification agreement does not establish factual truth or generator quality']}
    for name, root in roots.items():
        rows = list(read_rows(root / 'test-predictions.jsonl'))
        predictions[name] = {r['id']: r for r in rows}
        if len(predictions[name]) != len(rows):
            raise ValueError('Duplicate prediction IDs')
        metrics = {}
        for scale in ('raw_T1', 'calibrated'):
            groups = defaultdict(list)
            for r in rows:
                p = r[scale + '_yes_probability']
                if not math.isfinite(p) or not 0 <= p <= 1 or r['truncated']:
                    raise ValueError('Invalid prediction')
                groups[r['task']].append((p, r['label'] == 0))
            metrics[scale] = {}
            for task, pairs in groups.items():
                tp = sum(p >= .5 and y for p, y in pairs)
                tn = sum(p < .5 and not y for p, y in pairs)
                fp = sum(p >= .5 and not y for p, y in pairs)
                fn = sum(p < .5 and y for p, y in pairs)
                declared = reports[name]['splits']['test'][scale][task]
                for key, value in dict(tp=tp, tn=tn, fp=fp, fn=fn).items():
                    if declared[key] != value:
                        raise ValueError(f'{name}/{task}: incorrect {key}')
                metrics[scale][task] = {**declared, 'accuracy_ci': proportion(tp + tn, len(pairs)),
                                        'recall_ci': proportion(tp, tp + fn),
                                        'false_positive_rate_ci': proportion(fp, fp + tn)}
        result['models'][name] = {'checkpoint': reports[name]['model'], 'test': metrics,
                                  'temperature': reports[name]['temperature'],
                                  'report_sha256': sha256_file(root / 'report.json'),
                                  'predictions_sha256': sha256_file(root / 'test-predictions.jsonl')}
    if predictions['baseline'].keys() != predictions['candidate'].keys():
        raise ValueError('Unmatched test IDs')
    paired = defaultdict(lambda: dict(baseline_only_correct=0, candidate_only_correct=0, both_correct=0, both_wrong=0))
    for key, before in predictions['baseline'].items():
        after = predictions['candidate'][key]
        if (before['label'], before['task']) != (after['label'], after['task']):
            raise ValueError('Reference labels changed')
        y = before['label'] == 0
        a = (before['raw_T1_yes_probability'] >= .5) == y
        b = (after['raw_T1_yes_probability'] >= .5) == y
        category = ('both_correct' if b else 'baseline_only_correct') if a else ('candidate_only_correct' if b else 'both_wrong')
        paired[before['task']][category] += 1
    result['paired_changes'] = dict(paired)
    result['independent_recount_passed'] = True
    json_write(args.out, result)
    print(json.dumps(result['paired_changes'], indent=2))


if __name__ == '__main__':
    main()
