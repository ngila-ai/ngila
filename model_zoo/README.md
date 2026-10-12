# Model zoo

Registry: [registry.json](registry.json). `python -m ngila zoo` prints current status.

| Model | Status |
|---|---|
| Qwen3.5-9B | Original weights; six-prompt CPU smoke only |
| Qwen3.5-9B-Base | Local reference, not evaluated |
| Qwen3.5-0.8B | Engineering checks; tokenizer reconstruction unverified |
| ngila 9B v0.1 | Planned, no trained checkpoint or HF upload |

A release must identify its parent revision, intervention, discovery/tuning data
hashes, generation configuration, evaluation report, capability/calibration results,
license, and actual weight/adapter files. Steering vectors are custom runtime
artifacts, not standalone models or RLCD checkpoints. A direction fitted on 0.8B
arithmetic is not a censorship intervention and is not a release candidate.

Destination: Hugging Face `ngila-ai`. The publisher validates a staged `release.json`
with repo_type=model, base_model, evaluation_report, and a SHA256 inventory including
an actual safetensors file and model card. Model export remains pending a validated
candidate; no placeholder repository is advertised as a trained model.
