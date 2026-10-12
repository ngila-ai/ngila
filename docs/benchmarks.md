# Benchmark sources

Nine source adapters are registered. Three sources have been acquired and built; the other six are catalogued, not benchmarked.

| Source | License | What its labels measure | Acquired |
|---|---|---|---|
| [XSTest](https://github.com/paul-rottger/xstest) | cc-by-4.0 | overrefusal_and_matched_harmful_controls | yes |
| [SimpST](https://github.com/bertiev/SimpleSafetyTests) | cc-by-4.0 | harmful_prompt_response_behavior | yes |
| [ToxiC](https://huggingface.co/datasets/lmsys/toxic-chat) | cc-by-nc-4.0 | input_toxicity_and_jailbreak_labels | pending |
| [OpenAIMod](https://github.com/openai/moderation-api-release) | mit | moderation_classifier_test | pending |
| [Aegis](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0) | cc-by-4.0 | dialogue_moderation_and_prompt_subset | pending |
| [Aegis2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | cc-by-4.0 | separate_prompt_and_response_moderation | pending |
| [WildG](https://huggingface.co/datasets/allenai/wildguardmix) | odc-by | prompt_harm_response_harm_response_refusal | pending |
| [SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | cc-by-nc-4.0 | response_safety_and_helpfulness_preferences | pending |
| [Beavertails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | cc-by-nc-4.0 | QA_pair_moderation_and_prompt_pool | yes |

Pinned revisions, aliases and adapter notes: [source registry](../configs/sources.json).

ToxiC means ToxicChat here; SimpST means SimpleSafetyTests; WildG means WildGuardMix.
XSTest and SimpleSafetyTests remain test-only. OpenAIMod is arbitrary text classification, not a set of instructions for generation. Toxicity, response harmfulness, request completion, refusal and factuality are separate measurements.

BeaverTails labels QA pairs, so its standalone prompts retain unknown prompt risk. SafeRLHF helpfulness and safety preferences are distinct; reversing safety preferences does not establish a correct answer. Aegis annotator disagreement is retained. Aegis2 redacted content is skipped. WildGuard access terms require upstream acceptance; its gated data are not configured for republication.

## Built v0.1

- XSTest: 450 test records; SimpleSafetyTests: 100 test records.
- BeaverTails 30k: 17,232 train / 1,797 validation / 3,021 test records after grouping.
- 8,157 original training records overlap held-out prompt groups and are quarantined.
- Combined: 17,232 train / 1,797 validation / 3,571 test. Counts are records, not unique prompts.
- These sources are mainly English. Add Chinese factual, uncertainty, false-premise and capability holdouts before claiming success on censorship or RLCD readiness.

## Distribution

`ngila-ai/ngila-benchmarks-v0.1`: XSTest + SimpleSafetyTests, CC BY 4.0.
`ngila-ai/ngila-beavertails-v0.1`: BeaverTails, CC BY-NC 4.0.
Both datasets are published on Hugging Face; pinned upload commits are recorded in [publication status](publication-status.json). Licenses remain separate; normalization does not remove the NC restriction.

Fetch further sources with `python -m ngila fetch --source aegis2 --output data/downloads/aegis2`.
The receipt pins the resolved upstream revision. Supply the acquired files and their official roles in a new build config; no automatic train/test role inference.
