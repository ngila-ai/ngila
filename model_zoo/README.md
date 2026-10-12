# Model zoo

Registry: [registry.json](registry.json). `python -m ngila zoo` prints current status.

| Model | Status |
|---|---|
| Qwen3.5-9B | Original weights; six-prompt CPU smoke only |
| Qwen3.5-9B-Base | Local reference, not evaluated |
| Qwen3.5-0.8B | Engineering checks; tokenizer reconstruction unverified |
| ngila 9B v0.1 | Planned, no trained checkpoint or HF upload |
| S1-9B judge v1 | Local trained/calibrated candidate; not accepted, no HF upload |

S1 judge v1 classifies refusal and response harmfulness. Its matched held-out
results are in `eval/reports/s1-judge-supervised-v1.json`: refusal agreement 94.9%,
harmfulness agreement 90.6%, but harmful-response recall only 35/52. Fulfillment,
Chinese and long inputs are unvalidated. Weights live at
`runs/s1-judge-v1/calibrated/`; upstream S1 redistribution licensing has not yet
been verified. This classifier is separate from a generative ngila checkpoint.

A release must identify its parent revision, intervention, discovery/tuning data
hashes, generation configuration, evaluation report, capability/calibration results,
license, and actual weight/adapter files. Steering vectors are custom runtime
artifacts, not standalone models or RLCD checkpoints. A direction fitted on 0.8B
arithmetic is not a censorship intervention and is not a release candidate.

Destination: Hugging Face `ngila-ai`. The publisher validates a staged `release.json`
with repo_type=model, base_model, evaluation_report, and a SHA256 inventory including
an actual safetensors file and model card. Model export remains pending a validated
candidate; no placeholder repository is advertised as a trained model.
