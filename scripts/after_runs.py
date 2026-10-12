"""Wait for explicitly named generation manifests, then execute a pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifests', nargs='+', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--timeout', type=int, default=7200)
    a = p.parse_args()
    deadline = time.monotonic() + a.timeout
    while True:
        statuses = []
        for name in a.manifests:
            try:
                statuses.append(json.loads(Path(name).read_text())['status'])
            except (FileNotFoundError, json.JSONDecodeError):
                statuses.append('pending')
        if any(s == 'failed' for s in statuses):
            raise RuntimeError('A prerequisite generation failed')
        if all(s == 'complete' for s in statuses):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Prerequisite generation did not complete')
        time.sleep(10)
    raise SystemExit(subprocess.call([sys.executable, '-m', 'ngila', 'pipeline',
                                     '--config', a.config, '--output', a.output]))


if __name__ == '__main__':
    main()
