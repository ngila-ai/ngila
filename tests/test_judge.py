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
