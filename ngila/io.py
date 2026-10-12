import csv
import gzip
import hashlib
import json
import unicodedata
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prompt_key(text):
    return text_hash(' '.join(unicodedata.normalize('NFKC', text).casefold().split()))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def jsonl_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(path):
    path = Path(path)
    if path.suffix == '.arrow':
        from datasets import Dataset
        yield from Dataset.from_file(str(path))
    elif path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as f:
            yield from csv.DictReader(f)
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            if path.name.endswith('.json'):
                value = json.load(f)
                if not isinstance(value, list):
                    raise ValueError(f'Expected JSON list: {path}')
                yield from value
            else:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
