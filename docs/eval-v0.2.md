# Evaluation inventory v0.2

This supersedes the 550-prompt v0.1 export. Counts below were measured from pinned downloaded files, not copied from approximate names or papers.

**17 acquired sources; 147,037 held-out records; 143,438 normalized distinct inputs.** The latter includes moderation text, not only generation prompts. SORRY-Bench is the 18th registered source and remains gated/unacquired.

| Source | Test records retained | Unique generation/MC prompts | Use / limitation |
|---|---:|---:|---|
| [Aegis](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | 1,199 | 356 | dialogue_moderation_and_prompt_subset |
| [Aegis2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | 1,928 | 1,915 | separate_prompt_and_response_moderation |
| [Beavertails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | 3,021 | 2,525 | QA_pair_moderation_and_prompt_pool |
| [Do-Not-Answer](https://huggingface.co/datasets/LibrAI/do-not-answer) | 939 | 938 | refusal_and_unanswerable_taxonomy |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 1,319 | 1,319 | math_capability_regression |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | 320 | 320 | behavior_completion |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | 14,042 | 13,937 | broad_capability_and_decision_calibration |
| [OpenAIMod](https://github.com/openai/moderation-api-release) | 1,680 | 0 | moderation_classifier_test |
| [OR-Bench](https://huggingface.co/datasets/bench-llm/or-bench) | 82,333 | 81,015 | overrefusal_large_and_hard_controls |
| [SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | 8,211 | 7,494 | response_safety_and_helpfulness_preferences |
| [SafetyBench](https://huggingface.co/datasets/thu-coai/SafetyBench) | 22,870 | 22,844 | bilingual_safety_knowledge |
| [SimpleQA Verified](https://huggingface.co/datasets/google/simpleqa-verified) | 1,000 | 1,000 | short_factuality_and_abstention |
| [SimpST](https://github.com/bertiev/SimpleSafetyTests) | 100 | 100 | harmful_prompt_response_behavior |
| [ToxiC](https://huggingface.co/datasets/lmsys/toxic-chat) | 5,083 | 4,951 | input_toxicity_and_jailbreak_labels |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | 817 | 817 | truthfulness_and_false_beliefs |
| [WildG](https://huggingface.co/datasets/allenai/wildguardmix) | 1,725 | 1,725 | prompt_harm_response_harm_response_refusal |
| [XSTest](https://github.com/paul-rottger/xstest) | 450 | 450 | overrefusal_and_matched_harmful_controls |

## Evaluation tracks

| Track | Distinct prompts | Purpose |
|---|---:|---|
| core-behavior | 22,001 | All acquired behavioral suites plus OR-Bench hard/toxic; no random downsampling. |
| extended-behavior | 101,042 | Full behavioral pool including all OR-Bench 80k inputs; overlaps core. |
| factuality | 1,817 | Closed-book factuality; correct/incorrect/abstained scored separately with references. |
| math | 1,319 | Official math test, independent capability regression. |
| knowledge | 13,937 | Full MMLU official test; capability and explicit decision-probability calibration. |
| bilingual-safety | 22,844 | Parallel Chinese/English safety MC questions; test keys withheld, external accuracy evaluation required. |

Core and extended behavior overlap and must not be summed. Official record-weighted scoring should map outputs back through source_record_ids; generation deduplication is an optimization, not a change to benchmark weights. Normalized duplicates may have conflicting annotations; retain all references and adjudicate instead of silently selecting one.

## Important corrections

- OR-Bench pinned files contain 80,359 / 1,319 / 655 records (80k / hard / toxic). Hard overlaps the full pool. The source considers non-toxic subsets benign; this is a benchmark label, not a guarantee about every question.
- Aegis1: 1,199 records, only 359 user_message rows (356 distinct prompts). Response and multi-turn text are moderation inputs, not standalone generation prompts.
- Aegis2: 1,964 source test records; 36 empty/redacted prompts excluded, leaving 1,928. No reconstruction attempted.
- SafeRLHF: all three official test partitions, 3,036 + 2,848 + 2,327 = 8,211 response-pair records. Overlapping prompts deduplicated for generation.
- SafetyBench: 11,435 Chinese + 11,435 English records are parallel versions, not twice as many underlying questions. Official test labels are withheld; do not report local accuracy. Uses the full Chinese test, not the filtered subset.
- TruthfulQA: the 817 generation/MC variants are the same questions; this build imports generation references once. The official split is called validation; ngila reserves it as test.
- Do-Not-Answer: source card metadata says Apache but its data-license paragraph specifies CC BY-NC-SA 4.0. The data are packaged under the latter, separately. Its HF train split is an evaluation benchmark, not ngila training data.
- WildGuardTest was downloaded with existing access, without accepting new terms. It is local-only; the ngila HF release does not redistribute the gated data.
- SORRY-Bench (45 categories, 450 base prompts and Chinese variants) is registered but blocked by GatedRepoError. Its custom license prohibits redistribution, so future authorized use will also be local-only.
- HarmBench imports 320 official text test cases, retains contextual passages, and keeps copyright-overlap grading separate. The downloaded 80 text validation cases are not counted as test.

## Training contamination audit

The expanded holdouts quarantine 10,581 BeaverTails training records (2,424 more than v0.1). Revised train/validation counts: 15,039 / 1,566. Do not train from v0.1 and then claim clean v0.2 evaluation. No ngila checkpoint has been trained on either release. The older published datasets remain immutable historical versions. Exact normalization does not detect paraphrase or pretraining contamination.

## Why these additions matter

OR-Bench measures over-refusal directly; HarmBench measures substantive completion rather than a refusal keyword. SimpleQA Verified provides 1,000 rechecked factual questions and evidence URLs; TruthfulQA tests common false beliefs. GSM8K and MMLU provide objective capability controls. SafetyBench adds Chinese/English coverage but tests safety knowledge, not factual political censorship. Confidence calibration requires explicit event probabilities plus correctness labels; merely downloading these sets does not measure calibration.

Missing coverage remains: a sourced bilingual political/history factuality suite, controlled false-premise/sycophancy pairs, and evaluated uncertainty/calibration outputs. These cannot be replaced by more unsafe-prompt examples.

## Reproduce

```bash
python scripts/acquire_eval.py --sources toxicchat aegis1 aegis2 wildguard saferlhf openaimod orbench harmbench truthfulqa gsm8k safetybench donotanswer simpleqa_verified mmlu
python -m ngila pipeline --config configs/pipeline-v0.2.json --output runs/pipeline-v0.2
```

The build also uses the original local v0.1 source files, including the pinned BeaverTails Arrow cache. Adjust Python/cache paths on a different host. Acquisition records exact revisions and errors, never automatically accepts gates. The pipeline rebuilds data and exports tracks, not model answers. Use a new versioned output directory for reruns.

Generation example (not yet executed):

```bash
python scripts/experiment.py --model /mnt/data/models/Qwen/Qwen3.5-9B --device cuda --prompts data/processed/v0.2/suites/core-behavior.jsonl --output runs/9b-core-v0.2 --max-new-tokens 1024
```

Gold references remain in normalized data and must not be passed as model prompts. Review coverage, refusals, completion, correctness, harmfulness and truncation separately. No v0.2 model score has been produced.

See [machine inventory](eval-inventory-v0.2.json), [acquisition receipts](acquisition-v0.2.json), [dataset manifest](dataset-build-v0.2.json), and [release receipts](releases-v0.2.json).


## Additional candidates reviewed, not included in v0.2 counts

- [PHTest / FalseRefusal](https://github.com/umd-huang-lab/FalseRefusal): targeted pseudo-harmful false-refusal patterns with separate controversial labels; useful as an independent check beyond OR-Bench. Requires source/license review and an overlap audit before import.
- [ChineseSafetyQA](https://github.com/LivingFutureLab/ChineseSafetyQA): short factual Chinese safety-domain QA and MC formats; potentially more useful than unlabeled SafetyBench for local Chinese factuality scoring. It still does not replace a political/history gold set. Not downloaded into this release.
- [Chinese Refusals Benchmark](https://huggingface.co/datasets/joaocarloscruz/chinese-refusals-benchmark): 500 stratified prompts derived from QuixiAI/china-refusals. Relevant to political-topic behavior, but it is not an independent verified factual answer key. Audit overlap with discovery/training corpora, neutral wording, topic balance and reference evidence before adoption.

These candidates are research leads, not completed adapters, acquired datasets or evaluated model results.

## Published v0.2 dataset packages

| Repository | Test records | License |
|---|---:|---|
| [ngila-benchmarks-ccby-v0.2](https://huggingface.co/datasets/ngila-ai/ngila-benchmarks-ccby-v0.2) | 86,010 | cc-by-4.0 |
| [ngila-benchmarks-nc-v0.2](https://huggingface.co/datasets/ngila-ai/ngila-benchmarks-nc-v0.2) | 16,315 | cc-by-nc-4.0 |
| [ngila-benchmarks-mit-v0.2](https://huggingface.co/datasets/ngila-ai/ngila-benchmarks-mit-v0.2) | 41,231 | mit |
| [ngila-truthfulqa-v0.2](https://huggingface.co/datasets/ngila-ai/ngila-truthfulqa-v0.2) | 817 | apache-2.0 |
| [ngila-do-not-answer-v0.2](https://huggingface.co/datasets/ngila-ai/ngila-do-not-answer-v0.2) | 939 | cc-by-nc-sa-4.0 |

These packages contain 145,312 test records from 16 redistributable sources.
The 1,725 WildGuard test records remain local-only, making the local total 147,037.
The NC package also contains the revised 15,039 train and 1,566 validation records.
