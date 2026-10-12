"""Export explicit evaluation tracks; count records separately from unique inputs."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import export_prompts
from .io import json_write, read_rows, sha256_file


def prepare(dataset, config_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text())
    by_source = defaultdict(list)
    inputs = set()
    for row in read_rows(Path(dataset)/'test.jsonl'):
        by_source[row['source']].append(row)
        inputs.add(row['prompt_key'])
    summary = {}
    for source, rows in sorted(by_source.items()):
        eligible = [r for r in rows if r['prompt'] is not None]
        summary[source] = {
            'records':len(rows), 'unique_inputs':len({r['prompt_key'] for r in rows}),
            'unique_prompts':len({r['prompt_key'] for r in eligible}),
            'tasks':dict(Counter(r['task'] for r in rows)),
            'languages':dict(Counter(r['language'] for r in rows)),
            'source_splits':dict(Counter(r['source_split'] for r in rows)),
            'label_scopes':dict(Counter(r['label_scope'] for r in rows)),
        }
    tracks = {}
    for name, track in config['tracks'].items():
        tracks[name] = export_prompts(dataset,output/(name+'.jsonl'),sources=track['sources'],
                                     exclude_source_splits=track.get('exclude_source_splits'),
                                     tasks=tuple(track.get('tasks',['generation'])))
        tracks[name]['description'] = track['description']
    manifest = {'version':config['version'],'config_sha256':sha256_file(config_path),
                'dataset_manifest_sha256':sha256_file(Path(dataset)/'manifest.json'),
                'test_records':sum(v['records'] for v in summary.values()),
                'test_unique_inputs':len(inputs),'by_source':summary,'tracks':tracks,
                'limitations':['Tracks may overlap; never sum their sizes as independent questions',
                               'Metadata and answer keys are not part of the rendered model prompt',
                               'Downloaded/prepared is not generated/reviewed/scored',
                               'SafetyBench test keys withheld; accuracy requires official submission',
                               'No semantic deduplication or political factuality gold set yet']}
    json_write(output/'manifest.json',manifest)
    return manifest
