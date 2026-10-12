import json
import tempfile
import unittest
from pathlib import Path

from ngila.data import adapt, export_prompts
from ngila.hub import stage_dataset
from ngila.io import jsonl_write, json_write

REG={s['id']:s for s in json.loads(Path('configs/sources.json').read_text())['sources']}

class ExpandedSources(unittest.TestCase):
    def test_real_openai_schema_is_text_moderation(self):
        r,_=adapt({'prompt':'a quoted sentence','H':0},REG['openaimod'],'test',0,'rev')
        self.assertEqual(r['context'],'a quoted sentence')
        self.assertIsNone(r['prompt']);self.assertIsNone(r['source_labels']['V'])

    def test_repeated_aegis_conversation_id_keeps_distinct_rows(self):
        a,_=adapt({'id':'same','text':'question','text_type':'user_message'},REG['aegis1'],'test',0,'r')
        b,_=adapt({'id':'same','text':'answer','text_type':'llm_response'},REG['aegis1'],'test',1,'r')
        self.assertNotEqual(a['id'],b['id']);self.assertEqual(a['family_id'],b['family_id'])
        self.assertEqual(b['task'],'moderation')

    def test_harmbench_context_is_part_of_prompt(self):
        r,_=adapt({'Behavior':'Review the passage','ContextString':'A supplied passage','BehaviorID':'b','FunctionalCategory':'contextual','SemanticCategory':'category','Tags':''},REG['harmbench'],'test',0,'r')
        self.assertEqual(r['prompt'],'A supplied passage\n\nReview the passage')

    def test_mc_gold_not_in_prompt_and_parallel_family_preserved(self):
        r,_=adapt({'question':'Pick','choices':['one','two'],'answer':1,'subject':'math'},REG['mmlu'],'test',0,'r')
        self.assertEqual(r['source_labels']['answer_index'],1)
        self.assertEqual(r['prompt'],'Pick\n\nA. one\nB. two\n\nAnswer with the option letter only.')
        a,_=adapt({'id':12,'question':'q','options':['a','b'],'category':'c'},REG['safetybench'],'test_en',0,'r')
        b,_=adapt({'id':12,'question':'问题','options':['甲','乙'],'category':'c'},REG['safetybench'],'test_zh',0,'r')
        self.assertEqual(a['family_id'],b['family_id']);self.assertIsNone(a['source_labels']['answer_index'])
        self.assertEqual(b['language'],'zh');self.assertEqual(a['label_scope'],'withheld_answer_key')

    def test_orbench_overlap_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            rows=[adapt({'prompt':q,'category':'c'},REG['orbench'],split,i,'r')[0] for i,(q,split) in enumerate([
                ('same','or-bench-80k'),('same','or-bench-hard-1k'),('extra','or-bench-80k'),('control','or-bench-toxic')])]
            jsonl_write(p/'test.jsonl',rows);json_write(p/'manifest.json',{})
            full=export_prompts(p,p/'full.jsonl')
            core=export_prompts(p,p/'core.jsonl',exclude_source_splits=['or-bench-80k'])
            self.assertEqual(full['prompts'],3);self.assertEqual(core['prompts'],2)
            self.assertNotIn('source_labels',(p/'full.jsonl').read_text())

    def test_gated_dataset_cannot_accidentally_be_republished(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):stage_dataset(td,Path(td)/'release',['wildguard'],'ngila-ai/test')

    def test_dataset_license_overrides_misleading_card_tag(self):
        self.assertEqual(REG['donotanswer']['license'],'cc-by-nc-sa-4.0')

if __name__=='__main__':unittest.main()
