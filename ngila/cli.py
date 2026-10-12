import argparse
import json
from pathlib import Path

from . import data, evaluation, hub, pipeline, training
from .io import json_write, read_rows


def main():
    p=argparse.ArgumentParser(prog='ngila')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('catalog'); s.add_argument('--registry',default='configs/sources.json')
    s=sub.add_parser('fetch'); s.add_argument('--source',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('build'); s.add_argument('--config',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('prompts'); s.add_argument('--dataset',required=True); s.add_argument('--output',required=True)
    s.add_argument('--split',choices=['train','validation','test'],default='test'); s.add_argument('--sources',nargs='+'); s.add_argument('--limit',type=int,default=0)
    s=sub.add_parser('annotate'); s.add_argument('--outputs',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('eval'); s.add_argument('--outputs',required=True); s.add_argument('--annotations'); s.add_argument('--output',required=True)
    s=sub.add_parser('calibration'); s.add_argument('--predictions',required=True); s.add_argument('--output',required=True); s.add_argument('--bins',type=int,default=10)
    s=sub.add_parser('compare'); s.add_argument('--baseline',required=True); s.add_argument('--candidate',required=True)
    s.add_argument('--baseline-annotations',required=True); s.add_argument('--candidate-annotations',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('stage-dataset'); s.add_argument('--dataset',required=True); s.add_argument('--output',required=True)
    s.add_argument('--sources',nargs='+',required=True); s.add_argument('--repo-id',required=True)
    s=sub.add_parser('publish'); s.add_argument('--folder',required=True); s.add_argument('--receipt',required=True); s.add_argument('--private',action='store_true')
    s=sub.add_parser('pipeline'); s.add_argument('--config',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('report'); s.add_argument('--reports',nargs='+',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('zoo'); s.add_argument('--registry',default='model_zoo/registry.json')
    s=sub.add_parser('discovery'); s.add_argument('--outputs',required=True); s.add_argument('--annotations',required=True); s.add_argument('--output',required=True)
    s=sub.add_parser('train'); s.add_argument('--model',required=True); s.add_argument('--prompts',required=True); s.add_argument('--output',required=True)
    s.add_argument('--layer',type=int,required=True); s.add_argument('--python',required=True); s.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.command=='catalog':
        value=json.loads(Path(a.registry).read_text())
    elif a.command=='fetch': value=hub.fetch(a.source,a.output)
    elif a.command=='build': value=data.build(a.config,a.output)
    elif a.command=='prompts': value=data.export_prompts(a.dataset,a.output,a.split,a.sources,a.limit)
    elif a.command=='annotate': value=evaluation.annotation_template(a.outputs,a.output)
    elif a.command=='eval':
        value=evaluation.evaluate(a.outputs,a.annotations); json_write(a.output,value)
    elif a.command=='calibration':
        value=evaluation.calibration(read_rows(a.predictions),a.bins); json_write(a.output,value)
    elif a.command=='compare':
        value=evaluation.compare(a.baseline,a.candidate,a.baseline_annotations,a.candidate_annotations); json_write(a.output,value)
    elif a.command=='stage-dataset': value=hub.stage_dataset(a.dataset,a.output,a.sources,a.repo_id)
    elif a.command=='publish': value=hub.publish(a.folder,a.receipt,a.private)
    elif a.command=='pipeline': value=pipeline.run(a.config,a.output)
    elif a.command=='report': value=pipeline.render_report(a.reports,a.output)
    elif a.command=='zoo': value=json.loads(Path(a.registry).read_text())
    elif a.command=='discovery': value=training.discovery(a.outputs,a.annotations,a.output)
    elif a.command=='train': value=training.train(a.model,a.prompts,a.output,a.layer,a.python,a.device)
    print(json.dumps(value,ensure_ascii=False,indent=2))
