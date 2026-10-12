import math
import unittest
from ngila.judge import parse_response, QUESTIONS


class JudgeTests(unittest.TestCase):
    def response(self, p):
        return {'model': 'xiangxin-2.0.0', 'answers': {
            k: {'type': 'noul', 'noul': p} for k in QUESTIONS}}

    def test_probabilities_not_text_generation(self):
        decisions = parse_response(self.response(0.8), 0.9)
        self.assertEqual(decisions['response_unsafe'], {'probability': 0.8, 'label': False})

    def test_invalid_probabilities(self):
        for p in (True, None, '0.9', -0.1, 1.1, math.nan, math.inf):
            with self.subTest(p=p), self.assertRaises(ValueError):
                parse_response(self.response(p), 0.5)

    def test_missing_answer_is_not_safe(self):
        response = self.response(0.1)
        del response['answers']['response_unsafe']
        with self.assertRaises(KeyError):
            parse_response(response, 0.5)

    def test_model_required(self):
        response = self.response(0.1)
        del response['model']
        with self.assertRaises(ValueError):
            parse_response(response, 0.5)

    def test_no_auth_skips_key_and_preserves_incomplete(self):
        import io
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from ngila.judge import judge
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'outputs.jsonl'
            rows = [dict(id='a', text='4', prompt='2+2?', truncated=False, ended_with_eos=True),
                    dict(id='b', text='unfinished', prompt='why?', truncated=True, ended_with_eos=False)]
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            opener = MagicMock()
            opener.open.return_value = io.StringIO(json.dumps(self.response(0.1)))
            with patch.dict(os.environ, {'XIANGXIN_API_KEY': 'must-not-be-sent'}), \
                    patch('ngila.judge.build_opener', return_value=opener):
                report = judge(source, Path(tmp)/'result', endpoint='http://127.0.0.1:18739/v1/systemone', no_auth=True)
            request = opener.open.call_args.args[0]
            self.assertFalse(request.has_header('Authorization'))
            self.assertEqual(opener.open.call_count, 1)
            self.assertEqual(report['judged'], 1)
            self.assertEqual(report['skipped_incomplete'], 1)
            self.assertNotIn('must-not-be-sent', (Path(tmp)/'result/manifest.json').read_text())

    def test_malformed_response_object(self):
        for raw in (None, [], 'bad'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_response(raw, 0.5)
