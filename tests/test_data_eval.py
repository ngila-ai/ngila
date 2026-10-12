import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ngila.data import adapt, build, export_prompts
from ngila.evaluation import calibration, evaluate, compare
from ngila.hub import stage_dataset, validate_release
from ngila.io import json_write, jsonl_write, read_rows, text_hash
from ngila.pipeline import run
from ngila.training import discovery

REGISTRY = {s['id']:s for s in json.loads(Path('configs/sources.json').read_text())['sources']}


class AdapterTests(unittest.TestCase):
    def test_beavertails_labels_are_not_prompt_labels(self):
        r,_=adapt({'prompt':'a question','response':'an answer','category':{'politics':True},'is_safe':False},REGISTRY['beavertails'],'train',0,'rev')
        self.assertEqual(r['prompt_risk'],'unknown')
        self.assertEqual(r['label_scope'],'prompt_response_pair')
        self.assertIsNone(r['sft_target'])

    def test_aegis_dialogue_not_recast_as_prompt(self):
        r,_=adapt({'text':'mixed conversation','text_type':'combined'},REGISTRY['aegis1'],'train',0,'rev')
        self.assertEqual(r['task'],'moderation'); self.assertIsNone(r['prompt'])

    def test_redacted_aegis_skipped(self):
        r,why=adapt({'prompt':'REDACTED'},REGISTRY['aegis2'],'train',0,'rev')
        self.assertIsNone(r); self.assertEqual(why,'redacted_or_empty')

    def test_missing_openai_label_is_unknown(self):
        r,_=adapt({'text':'content','H':0},REGISTRY['openaimod'],'test',0,'rev')
        self.assertIsNone(r['source_labels']['S']); self.assertEqual(r['source_labels']['H'],0)

    def test_preferences_not_inverted(self):
        r,_=adapt({'prompt':'q','response_0':'a','response_1':'b','better_response_id':0,'safer_response_id':1},REGISTRY['saferlhf'],'train',0,'rev')
        self.assertEqual(r['source_labels']['better_response_id'],0)
        self.assertEqual(r['source_labels']['safer_response_id'],1)

    def test_wildguard_unknown_preserved(self):
        r,_=adapt({'prompt':'q','prompt_harm_label':None},REGISTRY['wildguard'],'test',0,'rev')
        self.assertEqual(r['prompt_risk'],'unknown')

    def test_toxicity_not_used_as_prompt_harm(self):
        r,_=adapt({'user_input':'q','toxicity':1},REGISTRY['toxicchat'],'train',0,'rev')
        self.assertEqual(r['prompt_risk'],'unknown')
        self.assertEqual(r['source_labels']['toxicity'],1)


class SplitsTests(unittest.TestCase):
    def test_cross_source_test_wins_and_train_prompt_grouping(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            jsonl_write(p/'train.jsonl',[{'prompt':q,'response':'a','category':{},'is_safe':True}
                                       for q in [' SAME  prompt ','other','other']])
            jsonl_write(p/'test.jsonl',[{'id':'1','prompt':'same prompt','label':'safe','type':'homonyms','focus':'x'}])
            config={'validation_fraction':0.5,'inputs':[
                {'source':'beavertails','path':str(p/'train.jsonl'),'split':'train','role':'train','revision':'r'},
                {'source':'xstest','path':str(p/'test.jsonl'),'split':'test','role':'test','revision':'r'}]}
            json_write(p/'config.json',config)
            m=build(p/'config.json',p/'built')
            self.assertEqual(m['quarantined'],1)
            self.assertEqual(m['counts']['test'],1)
            sets={s:{r['prompt_key'] for r in read_rows(p/'built'/f'{s}.jsonl')} for s in ('train','validation','test')}
            self.assertFalse(sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test'])
            self.assertIn(2,[m['counts']['train'],m['counts']['validation']])
            export_prompts(p/'built',p/'prompts.jsonl')
            self.assertEqual(len(list(read_rows(p/'prompts.jsonl'))),1)
            stage_dataset(p/'built',p/'release',['xstest'],'ngila-ai/test')
            validate_release(p/'release')
            (p/'release'/'secret.env').write_text('do not upload')
            with self.assertRaisesRegex(ValueError,'Unexpected'):
                validate_release(p/'release')

    def test_official_test_cannot_be_train(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            json_write(p/'config.json',{'inputs':[{'source':'xstest','path':'unused','split':'test','role':'train'}]})
            with self.assertRaisesRegex(ValueError,'test cannot'):
                build(p/'config.json',p/'out')


class MetricTests(unittest.TestCase):
    def test_truncation_and_unknown_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            rows=[{'id':'1','text':'I refuse','truncated':False,'ended_with_eos':True,'prompt_risk':'benign'},
                  {'id':'2','text':'unfinished','truncated':True,'ended_with_eos':False,'prompt_risk':'benign'},
                  {'id':'3','text':'no annotation','truncated':False,'ended_with_eos':True}]
            jsonl_write(p/'o.jsonl',rows)
            labels=[{'id':'1','output_sha256':text_hash('I refuse'),'behavior':'full_refusal','reviewer':'test'},
                    {'id':'2','output_sha256':text_hash('unfinished'),'behavior':'answer','reviewer':'test'}]
            jsonl_write(p/'a.jsonl',labels)
            r=evaluate(p/'o.jsonl',p/'a.jsonl')['overall']
            self.assertEqual(r['annotated_completed'],1); self.assertEqual(r['annotation_coverage'],0.5)
            self.assertEqual(r['benign_overrefusal_rate']['value'],1)
            self.assertIsNone(r['factual_correct_rate']['value'])
            self.assertEqual(evaluate(p/'o.jsonl')['status'],'awaiting_annotations')
            labels[0]['output_sha256']='stale'; jsonl_write(p/'a.jsonl',labels)
            with self.assertRaisesRegex(ValueError,'Stale'):
                evaluate(p/'o.jsonl',p/'a.jsonl')

    def test_known_calibration_values_and_unknown_label(self):
        r=calibration([{'id':'a','probability':0.8,'label':1},{'id':'b','probability':0.2,'label':0},
                       {'id':'c','probability':0.9,'label':None}])
        self.assertAlmostEqual(r['brier'],0.04); self.assertAlmostEqual(r['log_loss'],-__import__('math').log(0.8))
        self.assertEqual(r['unknown_labels'],1); self.assertAlmostEqual(r['ece'],0.2)
        with self.assertRaises(ValueError): calibration([{'id':'x','probability':float('nan'),'label':1}])

    def test_eval_data_cannot_feed_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            jsonl_write(p/'o.jsonl',[{'id':'1','split':'test','prompt':'q','text':'a','truncated':False,'ended_with_eos':True}])
            jsonl_write(p/'a.jsonl',[])
            with self.assertRaisesRegex(ValueError,'Only train'):
                discovery(p/'o.jsonl',p/'a.jsonl',p/'out.jsonl')


class PipelineTests(unittest.TestCase):
    def test_failure_recorded_and_following_stage_not_run(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            json_write(p/'config.json',{'id':'test','steps':[{'name':'fail','argv':['false']},{'name':'later','argv':['true']}]})
            with self.assertRaises(RuntimeError):
                run(p/'config.json',p/'run',p/'ledger.jsonl')
            record=json.loads((p/'run/run.json').read_text())
            self.assertEqual(record['status'],'failed'); self.assertEqual(len(record['steps']),1)
            self.assertEqual(len(list(read_rows(p/'ledger.jsonl'))),1)


if __name__=='__main__': unittest.main()
