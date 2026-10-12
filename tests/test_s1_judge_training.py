import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('prepare_s1', Path(__file__).resolve().parents[1] / 'scripts/prepare_s1_judge_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class S1LabelsTest(unittest.TestCase):
    def test_native_yes_no_order_and_unknowns(self):
        row = {'prompt': 'Example', 'response': 'Example response', 'subcategory': 'benign',
               'response_refusal_label': 'refusal', 'response_harm_label': 'unharmful'}
        rows = list(module.decisions(row, 'key', 1))
        self.assertEqual([r['label'] for r in rows], [0, 1])
        self.assertTrue(all([o['id'] for o in r['options']] == ['yes', 'no'] for r in rows))
        row['response_harm_label'] = None
        self.assertEqual([r['task'] for r in module.decisions(row, 'key', 1)], ['refusal'])
        row['response_refusal_label'] = None
        self.assertEqual(list(module.decisions(row, 'key', 1)), [])

    def test_split_is_prompt_grouped(self):
        from ngila.io import prompt_key
        self.assertEqual(module.split_for(prompt_key(' A  QUESTION ')), module.split_for(prompt_key('a question')))


if __name__ == '__main__':
    unittest.main()
