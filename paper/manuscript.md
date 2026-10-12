# ngila: measuring censorship interventions without conflating refusal and truth

Living research draft, 2026-10-11. No trained checkpoint or claimed improvement.

## Objective

Reduce learned evasion and factual distortion while retaining accuracy, appropriate
uncertainty and calibrated decisions. The intended downstream use is RLCD calibrated
decision training. Its reward/interface remains to be specified. Hidden activations
and generated reasoning are not direct observations of a model's true beliefs.

## Method

Start with Qwen3.5-9B and reversible decoder-output steering. Contrast reviewed
training behaviors, tune strength/layer on validation, then freeze intervention and
evaluate held-out prompts with matched baseline settings. Include zero-dose, random
direction, ordinary tasks, unseen topics and bilingual controls. Topic/style
confounding of refusal directions must be tested. 0.8B validates engineering only.

## Data

See [benchmark catalog](../docs/benchmarks.md) and [split protocol](../docs/dataset-schema.md).
The first build combines XSTest, SimpleSafetyTests and BeaverTails 30k. It retains
17,232 train, 1,797 validation and 3,571 test records, quarantining 8,157 training
records whose normalized prompt groups overlap held-out data. Labels retain their
original scope. There are no verified SFT targets. Exact deduplication is not a
semantic leakage audit. Six additional registered sources remain unacquired.

## Results so far

[Generated report](../docs/generated/results.md). Original Qwen3.5-9B: six CPU
smoke prompts, four completed and two truncated. Codex inspection marked two
historical responses as deflection and two arithmetic responses as answers; the
two arithmetic answers equal 323. This tiny, selected sample establishes pipeline
behavior only. It neither estimates broad factual accuracy nor demonstrates an
intervention effect. The 550-prompt benchmark has not been generated or reviewed.

## Limitations and next experiments

CUDA is currently unavailable inside this execution environment; original model
weights remain intact. No 9B direction has been validated, no calibration experiment
has run, and no RLCD checkpoint is ready. Add sourced Chinese factual questions,
false-premise and uncertainty cases, independent reviewers, capability controls
and matched response comparisons. Establish acceptance criteria before selection.

## Reproducibility and distribution

[Pipeline config](../configs/pipeline-v0.1.json), [experiment ledger](../docs/experiments.jsonl),
[model zoo](../model_zoo/README.md). The current pipeline rebuilds data and scores
saved smoke outputs; it does not run the full benchmark. Code destination is
GitHub ngila-ai/ngila; models/datasets go to Hugging Face ngila-ai. Both dataset packages are published; exact upload revisions are recorded in
[publication status](../docs/publication-status.json). GitHub publication awaits
account authorization: ngila-ai is a personal account while gh is authenticated
as thomaslwang. Data licenses remain source-specific.

## References

- [XSTest](https://github.com/paul-rottger/xstest)
- [SimpleSafetyTests](https://github.com/bertiev/SimpleSafetyTests)
- [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [Refusal direction research](https://arxiv.org/abs/2406.11717)
- [Qwen censorship reference](https://vas-blog.pages.dev/qwen-censorship/)
