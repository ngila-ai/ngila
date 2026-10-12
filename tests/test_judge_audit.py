import unittest
from ngila.judge_audit import score


class AuditTests(unittest.TestCase):
    def test_unknown_excluded_and_errors_counted(self):
        controls=[dict(id='a',expected_refusal=True,expected_fulfilled=False,expected_unsafe=None),
                  dict(id='b',expected_refusal=False,expected_fulfilled=True,expected_unsafe=True)]
        records=[dict(id='a',decisions={k:{'probability':.9} for k in ('refusal','request_fulfilled','response_unsafe')}),
                 dict(id='b',decisions={k:{'probability':.1} for k in ('refusal','request_fulfilled','response_unsafe')})]
        result=score(records,controls)
        self.assertEqual(result['fields']['refusal']['accuracy']['numerator'],2)
        self.assertEqual(result['fields']['request_fulfilled']['false_positives'],1)
        self.assertEqual(result['fields']['request_fulfilled']['false_negatives'],1)
        self.assertEqual(result['fields']['response_unsafe']['unknown_n'],1)
        self.assertEqual(result['fields']['response_unsafe']['accuracy']['n'],1)
        self.assertEqual(result['fields']['response_unsafe']['false_negatives'],1)

    def test_id_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            score([], [{'id':'a'}])
