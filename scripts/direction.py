"""Extract a unit mean-difference direction from discovery-only activations."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--positive', required=True)
    p.add_argument('--negative', required=True)
    p.add_argument('--layer', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.positive == a.negative:
        p.error('Contrasts must differ')
    m = json.loads((a.run / 'manifest.json').read_text())
    if m['status'] != 'complete' or m['args']['mode'] != 'collect':
        p.error('A completed collection run is required')
    batch = torch.load(a.run / 'activations.pt', map_location='cpu', weights_only=True)
    rows, h = batch['rows'], batch['residuals']
    if any(r['split'] != 'discovery' for r in rows):
        p.error('Direction extraction requires discovery-only prompts')
    if h.ndim != 3 or h.shape[0] != len(rows) or not 0 <= a.layer < h.shape[1]:
        p.error('Invalid activation shape or layer index')
    pos = [i for i, r in enumerate(rows) if r.get('group') == a.positive]
    neg = [i for i, r in enumerate(rows) if r.get('group') == a.negative]
    if min(len(pos), len(neg)) < 2:
        p.error('Each contrast needs at least two prompts; larger matched sets recommended')
    d = h[pos, a.layer].mean(0) - h[neg, a.layer].mean(0)
    norm = d.norm()
    if not torch.isfinite(d).all() or norm < 1e-8:
        p.error('Nonfinite or degenerate direction')
    artifact = {k: m[k] for k in ['model', 'config_sha256', 'template_sha256',
                                'weights_inventory', 'tokenizer_hashes']}
    artifact.update(direction=d / norm, layer=a.layer, positive=a.positive, negative=a.negative,
                    template=m['args']['template'], thinking=m['args']['thinking'],
                    positive_count=len(pos), negative_count=len(neg),
                    source_run=str(a.run.resolve()),
                    discovery_prompt_hashes=[hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        torch.save(artifact, f)
    print(f'Saved layer {a.layer} unit direction from {len(pos)} / {len(neg)} prompts')


if __name__ == '__main__':
    main()
