"""Acquire pinned registered v0.2 source files; never accept upstream gates."""
import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ngila.io import sha256_file, json_write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--registry',default='configs/sources.json')
    p.add_argument('--output',default='data/downloads/research-v0.2')
    p.add_argument('--sources',nargs='+')
    a=p.parse_args()
    sources=json.loads(Path(a.registry).read_text())['sources']
    statuses=[]
    for source in sources:
        if a.sources and source['id'] not in a.sources:continue
        if not source.get('files') or not source.get('revision'):continue
        dest=Path(a.output)/source['repo'].replace('/','--');dest.mkdir(parents=True,exist_ok=True)
        files,errors={},{}
        for name in source['files']:
            try:
                if source['provider']=='hf':
                    from huggingface_hub import hf_hub_download
                    path=Path(hf_hub_download(repo_id=source['repo'],repo_type='dataset',
                        filename=name,revision=source['revision'],local_dir=dest))
                else:
                    path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
                    url=f'https://raw.githubusercontent.com/{source["repo"]}/{source["revision"]}/{urllib.parse.quote(name)}'
                    with urllib.request.urlopen(url,timeout=60) as response:
                        payload=response.read()
                    path.write_bytes(payload)
                files[name]=sha256_file(path)
            except Exception as exc:errors[name]=type(exc).__name__
        receipt={'repo':source['repo'],'revision':source['revision'],'files':files,'errors':errors,
                 'status':'complete' if not errors else 'partial_or_blocked','gate_accepted':False}
        json_write(dest/'acquisition.json',receipt)
        statuses.append({'source':source['id'],'status':receipt['status'],'errors':errors})
        print(json.dumps(statuses[-1]),flush=True)
    if any(s['errors'] for s in statuses):raise SystemExit(1)

if __name__=='__main__':main()
