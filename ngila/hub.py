"""Pinned acquisition and allowlisted Hugging Face publication."""
import json
import shutil
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from .io import json_write, jsonl_write, read_rows, sha256_file


def fetch(source_id, output, registry_path='configs/sources.json'):
    registry = json.loads(Path(registry_path).read_text())
    source = next(s for s in registry['sources'] if s['id']==source_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        if source['provider'] == 'hf':
            from huggingface_hub import HfApi, snapshot_download
            info = HfApi().dataset_info(source['repo'], revision=source.get('revision'))
            revision = info.sha
            snapshot_download(repo_id=source['repo'], repo_type='dataset', revision=revision,
                              local_dir=output/'snapshot', cache_dir=output/'cache',
                              allow_patterns=source.get('files'))
        else:
            revision = source.get('revision')
            if not revision:
                url = f'https://api.github.com/repos/{source["repo"]}/commits/HEAD'
                with urllib.request.urlopen(url, timeout=15) as f:
                    revision = json.load(f)['sha']
            for name in source['files']:
                url = f'https://raw.githubusercontent.com/{source["repo"]}/{revision}/{urllib.parse.quote(name)}'
                path = output / 'snapshot' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(url, timeout=30) as response, path.open('wb') as f:
                    shutil.copyfileobj(response, f)
        files = {str(p.relative_to(output/'snapshot')):sha256_file(p)
                 for p in sorted((output/'snapshot').rglob('*')) if p.is_file() and '.cache' not in p.parts}
        manifest = {'status':'complete','source':source_id,'repo':source['repo'],
                    'revision':revision,'files':files,'license':source['license']}
    except Exception as e:
        # Avoid exception payloads that could include request headers or tokens.
        manifest = {'status':'blocked','source':source_id,'error_type':type(e).__name__,
                    'reason':'Check network, authentication and any upstream access gate; no gate accepted automatically.'}
        json_write(output/'fetch.json',manifest)
        raise RuntimeError(json.dumps(manifest)) from None
    json_write(output/'fetch.json',manifest)
    return manifest


def stage_dataset(dataset, destination, source_ids, repo_id, registry_path='configs/sources.json'):
    if not repo_id.startswith('ngila-ai/') or repo_id.count('/') != 1:
        raise ValueError('Publication namespace must be ngila-ai')
    registry = {s['id']:s for s in json.loads(Path(registry_path).read_text())['sources']}
    selected = [registry[k] for k in source_ids]
    licenses = {s['license'] for s in selected}
    if len(licenses) != 1:
        raise ValueError('Stage separate releases for different source licenses')
    if any(s.get('gated') or s.get('redistribute') is False for s in selected):
        raise ValueError('Gated-source redistribution is not configured; publish references instead')
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    counts, present, languages = {}, set(), set()
    for split in ('train','validation','test'):
        rows = [r for r in read_rows(Path(dataset)/f'{split}.jsonl') if r['source'] in source_ids]
        counts[split] = len(rows)
        present.update(r['source'] for r in rows)
        languages.update(r['language'] for r in rows if r['language'] not in ('unknown','mixed'))
        if rows:
            # Serialize variable taxonomies as JSON strings for a stable HF Arrow schema.
            for row in rows:
                for key in ('source_labels','reference_response'):
                    row[key + '_json'] = json.dumps(row.pop(key), ensure_ascii=False)
            jsonl_write(dest/f'{split}.jsonl', rows)
    if set(source_ids) != present:
        raise ValueError('Requested source has no records; refusing an incomplete release')
    license_name = next(iter(licenses))
    upstream = [{k:v for k,v in s.items() if k not in ('notes',)} for s in selected]
    manifest = json.loads((Path(dataset)/'manifest.json').read_text())
    version = manifest['config'].get('version','unversioned')
    inputs = [{k:v for k,v in i.items() if k!='path'} for i in manifest['inputs'] if i['source'] in source_ids]
    json_write(dest/'sources.json', {'sources':upstream, 'inputs':inputs})
    notices = ['# Source attribution', '', 'Normalized schema and grouped splits by ngila-ai. Original labels are unchanged.', '']
    for s in selected:
        notices += [f'- {s["id"]}: {s["url"]} — license {s["license"]}. Upstream authors retain attribution.']
        source_card = Path('data/downloads/research-v0.2') / s['repo'].replace('/','--') / 'README.md'
        if source_card.exists():
            target = dest/'upstream-cards'/f'{s["id"]}.md'; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_card,target)
        original_license = Path('data/raw') / s['id'] / 'LICENSE'
        if original_license.exists():
            target = dest/'licenses'/f'{s["id"]}.txt'; target.parent.mkdir(exist_ok=True)
            shutil.copyfile(original_license,target)
    notices += ['', 'License links:', '- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/',
                '- CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/',
                '- CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/',
                '- MIT: https://opensource.org/license/mit',
                '- Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0',
                'Source responses are historical model outputs, not verified factual answers or SFT targets.',
                'Original test roles preserved; duplicate train prompts that overlap held-out splits are excluded.',
                'Normalization preserves raw prompt text; NFKC/case/whitespace affect grouping only.']
    (dest/'NOTICE.md').write_text('\n'.join(notices)+'\n')
    configs = '\n'.join(f'      - split: {s}\n        path: {s}.jsonl' for s,n in counts.items() if n)
    language_yaml = '\n'.join('  - '+x for x in sorted(languages))
    card = f'''---
license: {license_name}
language:
{language_yaml}
task_categories:
  - text-classification
tags:
  - ngila
  - evaluation
  - refusal
configs:
  - config_name: default
    data_files:
{configs}
---
# {repo_id}

Version {version}: normalized research records for independent measurement of refusal,
over-refusal, factuality and calibration. This release does not provide a single
ngila score or verified training targets.

Counts are records, not unique prompts: {json.dumps(counts)}. Sources: {', '.join(source_ids)}.
Some evaluation suites use upstream splits named train or validation; they are
reserved as test here. Original split names remain in `source_split`.

See NOTICE.md for attribution and sources.json for pinned revisions and input hashes.
Source labels apply only at `label_scope`; do not transfer response/QA labels to prompts.
Factual/math/MC answer keys, where available, are evaluation references only.
SafetyBench test answer keys are withheld; local test accuracy cannot be computed.
`source_labels_json` and `reference_response_json` are JSON-encoded to preserve
heterogeneous upstream schemas. `prompt_risk=unknown` remains unknown.
`sft_target` is null; these are prompt/annotation research data, not approved SFT answers.

Official test records remain test-only. Training/validation splits are deterministic
prompt/family groups. Exact cross-source overlap is excluded from training;
semantic paraphrase leakage has not been audited. Do not train on the test split.

This dataset contains sensitive benchmark material. Published behavior labels are
not assertions of factual truth. Assess generation outputs independently.
'''
    (dest/'README.md').write_text(card)
    files = {str(p.relative_to(dest)):sha256_file(p) for p in dest.rglob('*') if p.is_file()}
    release = {'schema_version':1,'repo_type':'dataset','repo_id':repo_id,
               'status':'staged','counts':counts,'license':license_name,'files':files}
    json_write(dest/'release.json',release)
    return release


def validate_release(folder):
    folder = Path(folder)
    release = json.loads((folder/'release.json').read_text())
    if not release['repo_id'].startswith('ngila-ai/') or release['repo_id'].count('/') != 1:
        raise ValueError('Publication namespace must be ngila-ai')
    if release['repo_type'] not in ('dataset','model'):
        raise ValueError('Invalid repo type')
    expected = set(release['files']) | {'release.json'}
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unexpected or missing release files')
    for name, digest in release['files'].items():
        p = folder/name
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Release paths must be contained regular files')
        if sha256_file(p) != digest:
            raise ValueError('Release content changed after staging')
        if any(part.startswith('.') for part in Path(name).parts):
            raise ValueError('Hidden files are not publishable')
    if 'README.md' not in release['files']:
        raise ValueError('Missing model/dataset card')
    if release['repo_type']=='model':
        if not release.get('base_model') or not release.get('evaluation_report'):
            raise ValueError('Model release requires parent and evaluation provenance')
        if not any(name.endswith('.safetensors') for name in release['files']):
            raise ValueError('Model release contains no actual safetensors artifact')
    return release


def publish(folder, receipt, private=False):
    from huggingface_hub import HfApi
    release = validate_release(folder)
    try:
        api = HfApi()
        api.whoami()  # Do not log credential or identity payload.
        api.create_repo(repo_id=release['repo_id'],repo_type=release['repo_type'],
                        private=private,exist_ok=True)
        commit = api.upload_folder(repo_id=release['repo_id'], repo_type=release['repo_type'],
                                   folder_path=str(folder),
                                   allow_patterns=list(release['files'])+['release.json'],
                                   commit_message='Publish ngila versioned research artifacts')
        result = {'status':'published','repo_id':release['repo_id'],
                  'commit_url':commit.commit_url, 'commit_oid':commit.oid}
    except Exception as e:
        result = {'status':'blocked','repo_id':release['repo_id'],'error_type':type(e).__name__,
                  'reason':'Publication failed; check network and ngila-ai write access. No successful upload confirmed.'}
        json_write(receipt,result)
        raise RuntimeError(json.dumps(result)) from None
    json_write(receipt,result)
    return result
