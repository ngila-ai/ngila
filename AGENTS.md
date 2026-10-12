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


## Evaluation expansion v0.2

Current inventory: docs/eval-v0.2.md and docs/eval-inventory-v0.2.json.
17 acquired sources, 147,037 held-out records. Core behavior has 22,001 distinct
prompts; extended behavior 101,042 and overlaps core. Use configs/eval-suites-v0.2.json.
Keep factuality, capability, moderation, bilingual MC and refusal metrics separate.
SafetyBench test answer keys are withheld. WildGuard is acquired but not
redistributed; SORRY-Bench remains gated and prohibits redistribution.
Do-Not-Answer data use CC BY-NC-SA 4.0 despite Apache in HF metadata.
The v0.2 holdouts quarantine 10,581 original training records, leaving 15,039 train
and 1,566 validation. Do not use v0.1 training data for a clean v0.2 comparison.
Data acquisition and pipeline success are not model evaluation results.

## Evaluation priorities and judge (2026-10-12)

Two primary tracks: unsafe-prompt response behavior, and capability retention.
Use Terminal-Bench 2.0 first for capability; register other Qwen3.8-27B card
benchmarks for later. Preserve the historical v0.2 source archive.
The user selected Xiangxin (xiangxinai.cn), a System One decision model, as the
response judge. Use POST https://api.xiangxinai.cn/v1/systemone and Noul decisions,
not chat completion parsing. Pin xiangxin-2.0.0; record actual returned version.
Read XIANGXIN_API_KEY from the environment, never commit or print it.
Report response_unsafe, refusal and request fulfillment separately. The user
wants higher unsafe-response and lower refusal rates on unsafe prompts; neither
is a factual honesty score. Keep probabilities, raw responses and failed cases.

Preferred judge connection: existing Xiangxin deployment on t3090 via WireGuard
10.9.0.1, SSH port 50812, user tom. Remote service is loopback-only :18730;
forward local 127.0.0.1:18739 to remote 127.0.0.1:18730 and pass
`--endpoint http://127.0.0.1:18739/v1/systemone --no-auth` to `ngila judge`.
This route was verified and does not need the invalid public API credential.
Do not change/restart the shared inference service. Use low request concurrency.
