"""Prepare local-only supervised classification data; never preference inversion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngila.io import read_rows, prompt_key, sha256_file, json_write, jsonl_write
from ngila.judge import QUESTIONS_V2


def split_for(key):
    bucket = int(hashlib.sha256(('s1-judge-v1:' + key).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 70 else 'validation' if bucket < 80 else 'calibration' if bucket < 90 else 'test'


def decisions(row, key, index):
    state = 'User request:\n' + row['prompt'] + '\n\nAssistant response:\n' + row['response']
    mappings = {'refusal': ('response_refusal_label', {'refusal': 0, 'compliance': 1}),
                'response_unsafe': ('response_harm_label', {'harmful': 0, 'unharmful': 1})}
    for task, (field, labels) in mappings.items():
        if row.get(field) not in labels:
            continue
        yield {'id': f'wildguard-train:{index}:{task}', 'task': task, 'kind': 'noul',
               'state': state, 'question': QUESTIONS_V2[task]['instructions'],
               'api_question': QUESTIONS_V2[task],
               'options': [{'id': x, 'description': x} for x in ('yes', 'no')],
               'label': labels[row[field]], 'prompt_key': key,
               'source_row': index, 'source_label': row[field], 'category': row['subcategory']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude', action='append', required=True)
    ap.add_argument('--train-pairs', type=int, default=2048)
    ap.add_argument('--holdout-pairs', type=int, default=256)
    ap.add_argument('--max-state-tokens', type=int, default=1536)
    args = ap.parse_args()
    root = Path(args.out)
    if root.exists():
        raise FileExistsError(root)
    blocked = set()
    for path in args.exclude:
        for r in read_rows(path):
            if r.get('prompt_key'):
                blocked.add(r['prompt_key'])
            for field in ('prompt', 'user_prompt'):
                if isinstance(r.get(field), str):
                    blocked.add(prompt_key(r[field]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    candidates = []
    counts = Counter()
    for i, row in enumerate(read_rows(args.source)):
        if not isinstance(row.get('prompt'), str) or not isinstance(row.get('response'), str):
            counts['missing_prompt_or_response'] += 1
            continue
        key = prompt_key(row['prompt'])
        if key in blocked:
            counts['excluded_existing_evaluation_prompt'] += 1
            continue
        examples = list(decisions(row, key, i))
        if not examples:
            counts['unknown_response_labels'] += 1
            continue
        candidates.append((hashlib.sha256((key + row['response']).encode()).hexdigest(), key, examples))
    output = {s: [] for s in ('train', 'validation', 'calibration', 'test')}
    seen = set()
    for _, key, examples in sorted(candidates):
        if key in seen:
            counts['duplicate_prompt'] += 1
            continue
        seen.add(key)
        split = split_for(key)
        limit = args.train_pairs if split == 'train' else args.holdout_pairs
        if len(output[split]) >= limit:
            continue
        if len(tok.encode(examples[0]['state'], add_special_tokens=False)) > args.max_state_tokens:
            counts['over_token_limit'] += 1
            continue
        output[split].append(examples)
    root.mkdir(parents=True)
    files = {}
    for split, groups in output.items():
        if not groups:
            raise ValueError('Empty split: ' + split)
        rows = [r for group in groups for r in group]
        path = root / (split + '.jsonl')
        jsonl_write(path, rows)
        files[split] = {'pairs': len(groups), 'decisions': len(rows), 'sha256': sha256_file(path),
                        'labels': dict(Counter(f"{r['task']}:{r['label']}" for r in rows))}
    manifest = {'purpose': 'neutral_response_classification', 'source': 'allenai/wildguardmix',
                'revision': 'd29c47f41c8b51348b5c8e8c81c039b3132b66d1', 'source_split': 'train',
                'source_sha256': sha256_file(args.source), 'license': 'odc-by',
                'redistribute_raw_text': False, 'options_order': ['yes', 'no'],
                'label_zero_means': 'yes', 'rubric': 'ngila-xiangxin-v2',
                'split_unit': 'normalized prompt', 'selection': 'hash-ordered, one response per prompt',
                'exclusions': {p: sha256_file(p) for p in args.exclude},
                'max_state_tokens': args.max_state_tokens, 'counts': dict(counts), 'files': files,
                'preparation_args': vars(args), 'preparation_script_sha256': sha256_file(__file__),
                'limitations': ['English source; no bilingual generalization claim',
                    'Prompt-disjoint, not topic-disjoint', 'No request_fulfilled supervision',
                    'Upstream checkpoint training history may contain evaluation data; unseen status unverified',
                    'Length-filtered subset; not representative of long responses']}
    json_write(root / 'manifest.json', manifest)
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    main()
