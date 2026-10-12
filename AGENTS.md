# ngila — project memory

ngila is `align` backwards. This project researches model “anti-alignment”:
reducing learned censorship, evasions, and politically imposed factual distortion.
The user's goal is for models to answer according to their own knowledge rather
than produce imposed falsehoods. Operationally, evaluate evidence-based accuracy,
honest uncertainty, resistance to false premises, and calibrated decisions;
do not claim hidden activations or generated reasoning reveal a model's “true thoughts”.

Terminology: call project model artifacts **ngila checkpoints**, never
“anti-alignment checkpoints” / “反对齐 checkpoint”. ngila itself means anti-alignment.

The intended deliverable is a foundation checkpoint for **RLCD calibrated decision
training**. RLCD's exact training interface, reward definition, and acceptance
thresholds remain to be established; do not invent an expansion or substitute a
different RLCD algorithm. Zero refusal is not an adequate success metric.

First target: Qwen3.5-9B. Qwen3.5-0.8B is available for fast engineering tests,
not evidence that a direction, layer, or intervention transfers to 9B.
Qwen3.5-9B-Base is available as a reference; “foundation for RLCD” does not
necessarily mean the upstream Base checkpoint.

Local models: `/mnt/data/models/Qwen/{Qwen3.5-9B,Qwen3.5-9B-Base,Qwen3.5-0.8B}`.
Treat source weights as immutable. Keep runs and derived artifacts in this project.
Check GPU availability before each run; do not disturb other workloads.

Start with reversible activation steering and matched baseline runs. Distinguish
topic detection, refusal, response style, and factual accuracy. Do not extrapolate
one architecture's layer indices or coefficients to another. Separate direction
discovery, tuning, and held-out evaluation, including topic-level and bilingual
holdouts. Preserve raw outputs, truncation flags, settings, and provenance.
Do not use a single LLM judge, political agreement, or refusal keywords as truth.
Measure calibration and capability regressions before selecting an RLCD checkpoint.

References:
- https://vas-blog.pages.dev/qwen-censorship/
- https://github.com/Srinivasa314/qwen3.5-censorship
- https://github.com/p-e-w/heretic
- https://arxiv.org/abs/2406.11717

Current progress and reproducible commands are in README.md and docs/status.md.

## Project scope established 2026-10-11

Maintain five connected components: eval, versioned train/validation/test datasets,
reproducible train/eval pipeline, model zoo, and continuously updated paper/docs.
The user explicitly designated Hugging Face **ngila-ai** for models and datasets.
Publication there is authorized by the request; do not repeatedly seek permission.
Publish concrete, validated, attributed artifacts, never claim a remote repository
or a trained model exists without verification. Keep source licenses and gates.

Sources and aliases are in configs/sources.json. Preserve official test sets,
separate QA/response labels from prompt labels, retain unknown annotations, and
quarantine cross-split duplicate prompts. Do not merely invert safety preferences
into SFT answers: harmfulness and factual correctness are independent labels.
The current train implementation fits steering directions; it is not RLCD/SFT.
Keep test generation separate from discovery/training, record provenance, and
append experiments to docs/experiments.jsonl. Paper results must be backed by
actual report files and expose missing scores rather than fabricate them.

Code belongs in GitHub namespace **ngila-ai**, repository **ngila**.
Verified by GitHub API: ngila-ai is currently a personal account (User), not an
Organization. Authenticate for this project as ngila-ai using
`GH_CONFIG_DIR=/home/tom/.config/gh-ngila`; keep the other account unchanged.
Repository-local Git identity: ngila-ai <thomas.l.wang@gmail.com>.
GitHub repository is published at https://github.com/ngila-ai/ngila.
The user authorized repository publication there. Keep datasets/model weights on
Hugging Face ngila-ai, not in the Git repository.
