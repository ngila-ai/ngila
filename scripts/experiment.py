"""Local Qwen3.5 baseline, residual collection, and reversible steering."""
import argparse
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_prompts(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ('id', 'prompt', 'split')):
            raise ValueError('Each prompt requires nonempty string id, prompt, split')
        if row['id'] in ids:
            raise ValueError('Duplicate prompt id: ' + row['id'])
        ids.add(row['id'])
    if not rows:
        raise ValueError('Empty prompt set')
    return rows[:limit] if limit else rows


def steer_hook(direction, alpha):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden + alpha * direction.to(device=hidden.device, dtype=hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    return hook


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['generate', 'collect'], default='generate')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--template', choices=['chat', 'raw'], default='chat')
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--direction', type=Path)
    p.add_argument('--alpha', type=float, default=0.0)
    a = p.parse_args()
    if a.limit < 0 or a.max_new_tokens < 1 or a.threads < 1:
        p.error('limit >= 0, max-new-tokens >= 1, threads >= 1 required')
    if a.thinking and a.template != 'chat':
        p.error('--thinking requires chat template')
    if a.mode == 'collect' and a.direction:
        p.error('Collect only unmodified activations')
    if a.alpha and not a.direction:
        p.error('--alpha requires --direction')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable in this execution environment; no implicit CPU fallback')
    rows = load_prompts(a.prompts, a.limit)
    model_path = a.model.resolve()
    config_hash = digest(model_path / 'config.json')
    template_file = model_path / 'chat_template.jinja'
    tokenizer_file = model_path / 'tokenizer_config.json'
    template_hash = digest(template_file if template_file.exists() else tokenizer_file)
    files = sorted(model_path.glob('*.safetensors'))
    inventory = [{'name': x.name, 'size': x.stat().st_size,
                  'mtime_ns': x.stat().st_mtime_ns} for x in files]
    tokenizer_hashes = {name: digest(model_path / name) for name in
                        ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')
                        if (model_path / name).exists()}
    artifact = None
    if a.direction:
        artifact = torch.load(a.direction, map_location='cpu', weights_only=True)
        expected = {'model': str(model_path), 'config_sha256': config_hash,
                    'template': a.template, 'thinking': a.thinking,
                    'template_sha256': template_hash, 'weights_inventory': inventory,
                    'tokenizer_hashes': tokenizer_hashes}
        for key, value in expected.items():
            if artifact[key] != value:
                raise ValueError(f'Direction provenance mismatch: {key}')
        seen = set(artifact['discovery_prompt_hashes'])
        if any(prompt_hash(row['prompt']) in seen for row in rows):
            raise ValueError('Evaluation overlaps direction discovery prompts')
        if any(row['split'] == 'discovery' for row in rows):
            raise ValueError('Steering evaluation cannot use discovery split')
        vector = artifact['direction']
        if vector.ndim != 1 or not torch.isfinite(vector).all():
            raise ValueError('Invalid direction tensor')
        if not torch.isclose(vector.float().norm(), torch.tensor(1.), atol=1e-4):
            raise ValueError('Direction must have unit norm')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    dtype = torch.bfloat16 if a.device == 'cuda' else torch.float32
    metadata = {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'model': str(model_path), 'config_sha256': config_hash,
        'template_sha256': template_hash, 'prompts_sha256': digest(a.prompts),
        'script_sha256': digest(__file__),
        'weights_inventory': inventory, 'tokenizer_hashes': tokenizer_hashes,
        'weight_hash_note': 'Inventory only; full tensor file hashes not computed',
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'dtype': str(dtype),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
        'seed': 0, 'decoding': 'greedy', 'status': 'running',
        'direction_sha256': digest(a.direction) if a.direction else None,
        'hook_scope': 'all positions, prefill and decode; decoder block output',
    }
    manifest = a.output / 'manifest.json'
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    started = time.monotonic()
    handles = []
    try:
        if a.device == 'cpu':
            # Optional FLA kernels select CUDA even for CPU tensors in this version.
            # Use the model's own reference functions, only in this process.
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
                         'causal_conv1d_fn', 'causal_conv1d_update'):
                fn = getattr(qwen, name)
                setattr(qwen, name, inspect.unwrap(fn))
            metadata['linear_attention_backend'] = 'unwrapped upstream PyTorch reference (CPU)'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=dtype,
            device_map=a.device, attn_implementation='sdpa').eval()
        layers = model.model.language_model.layers
        captures = {}
        if a.mode == 'collect':
            def capture(index):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[index] = hidden[0, -1].detach().float().cpu().clone()
                return hook
            handles = [layer.register_forward_hook(capture(i)) for i, layer in enumerate(layers)]
        if artifact:
            layer = artifact['layer']
            if not 0 <= layer < len(layers) or vector.numel() != model.config.text_config.hidden_size:
                raise ValueError('Direction layer/dimension mismatch')
            handles.append(layers[layer].register_forward_hook(steer_hook(vector, a.alpha)))
        # Do not inherit sampling flags or forced output settings from a checkpoint.
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        eos_ids.add(tokenizer.eos_token_id)
        eos_ids.discard(None)
        gc = transformers.GenerationConfig(
            do_sample=False, max_new_tokens=a.max_new_tokens,
            eos_token_id=sorted(eos_ids),
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=True)
        metadata['generation_config'] = gc.to_dict()
        collected = []
        with (a.output / 'outputs.jsonl').open('w') as f, torch.inference_mode():
            for row in rows:
                rendered = row['prompt'] if a.template == 'raw' else tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': row['prompt']}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=a.thinking)
                inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to(a.device)
                t = time.monotonic()
                record = {**row, 'rendered_prompt': rendered,
                          'input_tokens': inputs.input_ids.shape[-1]}
                if a.mode == 'collect':
                    captures.clear()
                    model(**inputs, use_cache=False, logits_to_keep=1)
                    if len(captures) != len(layers):
                        raise RuntimeError('Missing decoder block captures')
                    collected.append(torch.stack([captures[i] for i in range(len(layers))]))
                else:
                    ids = model.generate(**inputs, generation_config=gc)[0, inputs.input_ids.shape[-1]:]
                    ended = bool(len(ids) and ids[-1].item() in eos_ids)
                    record.update(text=tokenizer.decode(ids, skip_special_tokens=False),
                                  generated_tokens=len(ids), ended_with_eos=ended,
                                  truncated=not ended and len(ids) >= a.max_new_tokens)
                record['seconds'] = time.monotonic() - t
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(json.dumps({'id': row['id'], 'seconds': record['seconds'],
                                  'text': record.get('text')}, ensure_ascii=False), flush=True)
        if collected:
            torch.save({'residuals': torch.stack(collected), 'rows': rows}, a.output / 'activations.pt')
        metadata['status'] = 'complete'
    except BaseException as e:
        metadata.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        for handle in handles:
            handle.remove()
        metadata['seconds'] = time.monotonic() - started
        manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
