import json
import tempfile
import unittest
from pathlib import Path
from ngila.io import jsonl_write, text_hash, read_rows
from ngila.training import judge_discovery


class JudgeDiscoveryTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(id=str(i), prompt=f'prompt {i}', text=f'response {i}', split='train',
                     truncated=False, ended_with_eos=True) for i in range(5)]
        decisions = []
        for i, r in enumerate(rows):
            p, f = ((0.95, 0.05) if i < 2 else (0.05, 0.95)) if i < 4 else (0.5, 0.5)
            decisions.append(dict(id=r['id'], output_sha256=text_hash(r['text']),
                prompt_sha256=text_hash(r['prompt']), status='ok', raw_response={'model':'xiangxin-2.0.0'},
                decisions={'refusal':{'probability':p}, 'request_fulfilled':{'probability':f}}))
        jsonl_write(root/'outputs.jsonl', rows)
        jsonl_write(root/'decisions.jsonl', decisions)
        return rows, decisions

    def test_uncertain_excluded_and_safety_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            result=judge_discovery(root/'outputs.jsonl',root/'decisions.jsonl',root/'discovery.jsonl')
            self.assertEqual(result['groups'], {'refusal':2,'answer':2})
            self.assertTrue(all(r['split']=='discovery' for r in read_rows(root/'discovery.jsonl')))

    def test_heldout_and_stale_rejected(self):
        for failure in ('heldout','stale'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); rows, decisions=self.fixture(root)
                if failure=='heldout': rows[0]['split']='validation'
                else: decisions[0]['prompt_sha256']='changed'
                jsonl_write(root/'outputs.jsonl',rows); jsonl_write(root/'decisions.jsonl',decisions)
                with self.assertRaises(ValueError):
                    judge_discovery(root/'outputs.jsonl',root/'decisions.jsonl',root/'discovery.jsonl')

    def test_incomplete_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rows,_=self.fixture(root)
            rows[0]['truncated']=True; jsonl_write(root/'outputs.jsonl',rows)
            with self.assertRaises(ValueError):
                judge_discovery(root/'outputs.jsonl',root/'decisions.jsonl',root/'discovery.jsonl')
