# Dataset protocol v0.1

Each normalized record preserves source, source revision, original split, raw prompt,
family ID, category, language, label scope, prompt risk, source labels and historical
reference responses. `sft_target` is null. Historical completions are not factual
answer keys. HF exports encode heterogeneous source labels and reference responses
as JSON strings for stable Arrow loading.

Official tests and evaluation-only suites cannot become training data. Group raw
prompts using Unicode NFKC, case folding and whitespace normalization, plus family
IDs. If a group overlaps partitions, test takes precedence over validation over
train; lower-priority records are quarantined. Remaining training groups are split
with a fixed seed and a 10% hash threshold for validation. This is not a guarantee
of exactly 10% of records. Within-split response variants may remain. Generation
exports deduplicate prompts and retain all source/record associations.

The manifest records input revisions, SHA256 hashes, output hashes, counts and
quarantine statistics. Exact grouping does not detect semantic paraphrases, shared
upstream pretraining, or topic leakage. Audit these before a publishable experiment.

`configs/data-local-v0.1.json` references the current machine's BeaverTails Arrow
cache. On another machine, fetch the same revision, point input paths to its files,
and preserve split names/roles. Never mix overlapping 30k and 330k BeaverTails pools.

Dataset release manifests allow only inventoried files with matching hashes.
Different source licenses are staged in separate repositories. No credentials,
local cache, raw model weights or unrelated workspace data belong in a release.
