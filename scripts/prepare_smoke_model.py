"""Make a local read-only-weight view, omitting an empty tokenizer.json.

Transformers reconstructs the tokenizer from the existing vocab and merges.
This is for engineering smoke tests, not a verified upstream tokenizer replacement.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
src = a.source.resolve()
if not (src / 'tokenizer.json').exists() or (src / 'tokenizer.json').stat().st_size != 0:
    p.error('Expected an existing empty tokenizer.json; otherwise use the original model')
for name in ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'config.json'):
    if not (src / name).exists() or not (src / name).stat().st_size:
        p.error(f'Missing fallback input: {name}')
a.output.mkdir(parents=True, exist_ok=False)
for f in src.iterdir():
    if f.is_file() and f.stat().st_size:
        (a.output / f.name).symlink_to(f)
(a.output / 'ngila-local-view.json').write_text(json.dumps({
    'source': str(src), 'omitted': ['tokenizer.json'],
    'reason': 'Empty tokenizer; reconstruct from original vocab/merges for smoke testing only'
}, indent=2))
print(a.output.resolve())
