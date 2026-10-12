"""Adapters keep upstream label scope; split by prompt/family across sources."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .io import json_write, jsonl_write, prompt_key, read_rows, sha256_file, text_hash


def safe_label(value):
    if value is None or value == '':
        return 'unknown'
    if value in ('safe', 'unharmful'):
        return 'benign'
    if value in ('unsafe', 'harmful'):
        return 'harmful'
    raise ValueError(f'Unrecognized safety label: {value!r}')


def adapt(row, source, split, index, revision):
    key = source['id']
    prompt, context = None, None
    risk, task, scope = 'unknown', 'generation', 'prompt'
    response = None
    categories = []
    labels = {}
    family = None
    if key == 'xstest':
        prompt = row['prompt']; risk = safe_label(row['label'])
        categories = [row['type']]
        family = row['type'].removeprefix('contrast_') + ':' + row.get('focus', '')
        labels = {'label': row['label'], 'type': row['type']}
    elif key == 'simplesafetytests':
        prompt = row['prompts_final']; risk = 'harmful'
        categories = [row['harm_area'], row['category']]
        labels = {'expected_by_source': 'refusal'}
    elif key == 'toxicchat':
        prompt, response = row['user_input'], row.get('model_output')
        labels = {k: row.get(k) for k in ['toxicity', 'jailbreaking', 'human_annotation']}
        # Toxicity is not equivalent to dangerous instruction or factual falsehood.
        scope = 'prompt_toxicity'
    elif key == 'openaimod':
        context = row.get('text', row.get('prompt')); task, scope = 'moderation', 'text'
        labels = {k: row.get(k) for k in ('S','H','V','HR','SH','S3','H2','V2')}
    elif key == 'aegis1':
        scope = 'dialogue'; task = 'moderation'; context = row['text']
        if row['text_type'] == 'user_message':
            prompt, context, task, scope = row['text'], None, 'generation', 'prompt'
        labels = {k: row.get(k) for k in ('text_type','num_annotations','labels_0','labels_1','labels_2','labels_3','labels_4')}
        # Preserve annotator disagreement; no ad hoc majority-vote definition.
    elif key == 'aegis2':
        prompt, response = row['prompt'], row.get('response')
        if not prompt or prompt.strip() == 'REDACTED':
            return None, 'redacted_or_empty'
        risk = safe_label(row.get('prompt_label'))
        labels = {k: row.get(k) for k in ('prompt_label','response_label','label_source','prompt_label_source','response_label_source')}
        categories = [c.strip() for c in (row.get('violated_categories') or '').split(',') if c.strip()]
        scope = 'prompt_and_response_separate'
    elif key == 'wildguard':
        prompt, response = row['prompt'], row.get('response')
        risk = safe_label(row.get('prompt_harm_label'))
        labels = {k: row.get(k) for k in ('prompt_harm_label','response_harm_label','response_refusal_label','adversarial',
                                        'prompt_harm_agreement','response_harm_agreement','response_refusal_agreement')}
        categories = [row['subcategory']] if row.get('subcategory') else []
        scope = 'prompt_and_response_separate'
    elif key == 'beavertails':
        prompt, response = row['prompt'], row['response']; scope = 'prompt_response_pair'
        labels = {'is_safe': row['is_safe'], 'category': row['category']}
        categories = [k for k, v in row['category'].items() if v is True]
    elif key == 'saferlhf':
        prompt = row['prompt']; scope = 'response_pair_preferences'
        response = {'response_0': row['response_0'], 'response_1': row['response_1']}
        labels = {k: v for k, v in row.items() if k not in ('prompt','response_0','response_1')}
        # better_response_id and safer_response_id remain independent; never invert.
    elif key == 'orbench':
        prompt = row['prompt']; categories = [row['category']]
        risk = 'harmful' if split == 'or-bench-toxic' else 'benign'
        labels = {'source_intended_risk':risk, 'subset':split}
    elif key == 'harmbench':
        context = row.get('ContextString') or None
        prompt = (context + '\n\n' if context else '') + row['Behavior']
        categories = [row['FunctionalCategory'], row['SemanticCategory']]
        risk = 'harmful'; family = row['BehaviorID']
        labels = {k:row[k] for k in ('Behavior','BehaviorID','FunctionalCategory','SemanticCategory','Tags')}
        labels['evaluation_protocol'] = 'copyright_overlap' if row['FunctionalCategory']=='copyright' else 'behavior_completion'
    elif key == 'truthfulqa':
        prompt = row['question']; scope = 'factual_reference'; categories = [row['category']]
        labels = {k:row[k] for k in ('type','best_answer','correct_answers','incorrect_answers','source')}
    elif key == 'gsm8k':
        prompt = row['question']; scope = 'math_reference'
        labels = {'solution':row['answer'], 'final_answer':row['answer'].rsplit('####',1)[-1].strip()}
    elif key == 'simpleqa_verified':
        prompt = row['problem']; scope = 'factual_reference'; categories = [row['topic']]
        labels = {k:v for k,v in row.items() if k!='problem'}
    elif key in ('mmlu','safetybench'):
        options = row['choices'] if key=='mmlu' else row['options']
        prompt = row['question'] + '\n\n' + '\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(options))
        prompt += '\n\n' + ('仅回答选项字母。' if split.endswith('_zh') else 'Answer with the option letter only.')
        task,scope = 'multiple_choice','answer_key' if 'answer' in row else 'withheld_answer_key'
        labels = {'options':options,'answer_index':row.get('answer')}
        categories = [row['subject'] if key=='mmlu' else row['category']]
        if key=='safetybench':family = str(row['id'])
    elif key == 'donotanswer':
        prompt = row['question']; scope = 'source_response_annotations'
        categories = [row[k] for k in ('risk_area','types_of_harm','specific_harms')]
        response = {k:v for k,v in row.items() if k.endswith('_response')}
        labels = {k:v for k,v in row.items() if k.endswith(('_harmful','_action'))}
        labels['source_expected_behavior'] = 'do_not_answer'
        # Includes unanswerable/false-premise requests; do not mark every prompt harmful.
    else:
        raise ValueError(f'No adapter implemented for {key}')
    text = prompt if prompt is not None else context
    if not isinstance(text, str) or not text.strip():
        return None, 'empty_text'
    origin_id = str(row.get('id', row.get('conv_id', index)))
    # Aegis repeats a conversation ID across user, response and combined rows.
    record_id = f'{origin_id}:{index}' if key == 'aegis1' else origin_id
    if key == 'aegis1' and row.get('id') is not None:
        family = origin_id
    record = {
        'schema_version': 1, 'id': f'{key}:{split}:{record_id}',
        'source': key, 'source_revision': revision, 'source_split': split,
        'source_id': origin_id, 'source_url': source['url'], 'license': source['license'],
        'task': task, 'prompt': prompt, 'context': context, 'prompt_key': prompt_key(text),
        'family_id': f'{key}:{family}' if family else None,
        'prompt_risk': risk, 'label_scope': scope, 'source_labels': labels,
        'categories': categories, 'reference_response': response,
        'language': ('zh' if split.endswith('_zh') else 'en') if key=='safetybench' else source.get('language', 'unknown'),
        'sft_target': None,
    }
    return record, None


def build(config_path, output, registry_path='configs/sources.json'):
    config = json.loads(Path(config_path).read_text())
    registry = {r['id']: r for r in json.loads(Path(registry_path).read_text())['sources']}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, inputs, skipped = [], [], Counter()
    seen_ids = set()
    for item in config['inputs']:
        source = registry[item['source']]
        role = item['role']
        if role not in ('train', 'validation', 'test'):
            raise ValueError('Invalid input role')
        if (source['eval_only'] or 'test' in item['split'].lower()) and role != 'test':
            raise ValueError('Official/eval-only test cannot become training or validation')
        inputs.append({**item, 'sha256': sha256_file(item['path'])})
        for i, row in enumerate(read_rows(item['path'])):
            record, reason = adapt(row, source, item['split'], i, item['revision'])
            if record is None:
                skipped[f'{source["id"]}:{reason}'] += 1
                continue
            if record['id'] in seen_ids:
                raise ValueError(f'Duplicate record id: {record["id"]}')
            seen_ids.add(record['id'])
            record['_role'] = role
            rows.append(record)
    if not rows:
        raise ValueError('No records imported')
    # Union exact normalized prompts across sources and supplied within-source families.
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    keys = {}
    for i, r in enumerate(rows):
        for k in [('prompt', r['prompt_key']), ('family', r['family_id'])]:
            if k[1] is None:
                continue
            if k in keys:
                parent[root(i)] = root(keys[k])
            else:
                keys[k] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    priority = {'train': 0, 'validation': 1, 'test': 2}
    partitions = defaultdict(list)
    validation_fraction = config.get('validation_fraction', 0.1)
    if not 0 <= validation_fraction < 1:
        raise ValueError('Invalid validation_fraction')
    quarantined = []
    for members in groups.values():
        protected = max((rows[i]['_role'] for i in members), key=priority.get)
        stable_key = min(rows[i]['prompt_key'] for i in members)
        bucket = int(text_hash(str(config.get('seed', 0)) + ':' + stable_key)[:16], 16) / 2**64
        partition = 'validation' if protected == 'train' and bucket < validation_fraction else protected
        for i in members:
            r = rows[i]
            original_role = r.pop('_role')
            r['split'] = partition
            if priority[original_role] < priority[protected]:
                r['quarantine_reason'] = 'overlap_with_' + protected
                quarantined.append(r)
            else:
                partitions[partition].append(r)
    output.mkdir(parents=True)
    for split in ('train', 'validation', 'test'):
        jsonl_write(output / f'{split}.jsonl', sorted(partitions[split], key=lambda x:x['id']))
    jsonl_write(output / 'quarantine.jsonl', quarantined)
    manifest = {'schema_version': 1, 'config': config, 'inputs': inputs,
                'registry_sha256': sha256_file(registry_path),
                'counts': {s:len(partitions[s]) for s in ('train','validation','test')},
                'quarantined': len(quarantined), 'skipped': dict(skipped),
                'source_counts': dict(Counter(r['source'] for v in partitions.values() for r in v)),
                'dedup': 'NFKC + casefold + whitespace; family union; held-out wins, conflicting train rows quarantined',
                'limitation': 'Semantic/paraphrase leakage is not detected',
                'files': {f'{s}.jsonl': sha256_file(output/f'{s}.jsonl') for s in ('train','validation','test','quarantine')}}
    json_write(output / 'manifest.json', manifest)
    return manifest


def export_prompts(dataset, output, split='test', sources=None, limit=0,
                   exclude_source_splits=None, tasks=('generation',)):
    if split not in ('train','validation','test') or limit < 0:
        raise ValueError('Invalid split or limit')
    items = defaultdict(list)
    for row in read_rows(Path(dataset) / f'{split}.jsonl'):
        if row['task'] not in tasks or (sources and row['source'] not in sources):
            continue
        if row['source_split'] in (exclude_source_splits or []):
            continue
        items[row['prompt_key']].append(row)
    result = []
    for key, group in sorted(items.items()):
        first = sorted(group, key=lambda x:x['id'])[0]
        risks = {r['prompt_risk'] for r in group}
        risk = next(iter(risks)) if len(risks) == 1 else 'unknown'
        result.append({'id':key, 'prompt':first['prompt'], 'split':split,
                       'source':first['source'], 'sources':sorted({r['source'] for r in group}),
                       'source_record_ids':[r['id'] for r in group], 'prompt_risk':risk,
                       'source_splits':sorted({r['source']+':'+r['source_split'] for r in group}),
                       'task':first['task'],
                       'source_prompt_risks':{r['id']:r['prompt_risk'] for r in group},
                       'categories':sorted({c for r in group for c in r['categories']}),
                       'language':first['language'],
                       'dataset_manifest_sha256':sha256_file(Path(dataset)/'manifest.json')})
    if limit:
        result = result[:limit]
    if not result:
        raise ValueError('No eligible prompts')
    if Path(output).exists():
        raise FileExistsError(output)
    jsonl_write(output, result)
    return {'prompts':len(result), 'split':split, 'deduplicated':True,
            'limited':bool(limit), 'sha256':sha256_file(output)}
