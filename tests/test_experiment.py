"""Check causal-intervention invariants without downloading a model."""
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('experiment', Path(__file__).parents[1] / 'scripts/experiment.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)
direction_spec = importlib.util.spec_from_file_location('direction', Path(__file__).parents[1] / 'scripts/direction.py')
direction = importlib.util.module_from_spec(direction_spec)
direction_spec.loader.exec_module(direction)


class SteeringTest(unittest.TestCase):
    def test_zero_dose_preserves_output(self):
        h = torch.randn(2, 4, 8)
        d = torch.randn(8)
        result = experiment.steer_hook(d, 0)(None, None, h)
        torch.testing.assert_close(result, h, rtol=0, atol=0)

    def test_only_requested_direction_changes(self):
        h = torch.randn(1, 5, 3)
        original = h.clone()
        d = torch.tensor([0., 1., 0.])
        cache = object()
        result, returned_cache = experiment.steer_hook(d, -2)(None, None, (h, cache))
        torch.testing.assert_close(result[..., 0], h[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(result[..., 2], h[..., 2], rtol=0, atol=0)
        torch.testing.assert_close(result[..., 1], h[..., 1] - 2)
        torch.testing.assert_close(h, original, rtol=0, atol=0)
        self.assertIs(cache, returned_cache)

    def test_duplicate_prompt_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'p.jsonl'
            row = json.dumps({'id': 'x', 'prompt': 'test', 'split': 'smoke'})
            path.write_text(row + '\n' + row)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                experiment.load_prompts(path, 0)

    def test_direction_extraction_and_split_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = dict(status='complete', args=dict(mode='collect', template='chat', thinking=False),
                            model='fixture', config_sha256='fixture', template_sha256='fixture',
                            weights_inventory=[], tokenizer_hashes={})
            (run / 'manifest.json').write_text(json.dumps(manifest))
            rows = [dict(id=str(i), prompt=str(i), split='discovery', group='p' if i < 2 else 'n')
                    for i in range(4)]
            h = torch.tensor([[[2., 1.]], [[2., 3.]], [[0., 1.]], [[0., 3.]]])
            torch.save(dict(rows=rows, residuals=h), run / 'activations.pt')
            argv = ['direction', '--run', str(run), '--positive', 'p', '--negative', 'n',
                    '--layer', '0', '--output', str(run / 'd.pt')]
            with patch('sys.argv', argv):
                direction.main()
            result = torch.load(run / 'd.pt', weights_only=True)
            torch.testing.assert_close(result['direction'], torch.tensor([1., 0.]))
            self.assertEqual(len(result['discovery_prompt_hashes']), 4)
            rows[-1]['split'] = 'test'
            torch.save(dict(rows=rows, residuals=h), run / 'activations.pt')
            with patch('sys.argv', argv), self.assertRaises(SystemExit) as error:
                direction.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
