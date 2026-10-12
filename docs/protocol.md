# Qwen3.5-9B study protocol (draft, not executed)

## Objective

Produce an RLCD training starting point with less learned evasion and factual
distortion, without sacrificing honest uncertainty or decision calibration.
Refusal suppression and factual improvement must be reported separately.

## Architecture and intervention

Local 9B config: 32 decoder blocks, hidden size 4096, three Gated DeltaNet blocks
followed by one full-attention block per group of four. The 0.8B model instead
has 24 blocks and hidden size 1024. Directions are checkpoint-specific.

Collect the residual at the last rendered prompt token, at every block output.
Direction = unit(mean(positive) - mean(negative)). Add alpha * direction at a
chosen block output on all positions, including prefill and subsequent decoding.
This is an additive intervention, not an orthogonal projection or permanent edit.
Block index is zero-based; blog tap numbers must be translated explicitly.

The three contrasts suggested by the reference blog (topic, refusal, style) are
hypotheses to test independently. A topic direction can remove topic information
as well as censorship; a style direction can merely switch between bad answers.
Do not interpret either as a truth direction.

## Data splits

- Discovery: matched prompts for extracting directions. Balance language,
  syntactic structure, length, and response format; preserve original outputs.
- Validation: select layers, doses, and combinations. Include multiple random
  unit directions at matching norms, zero dose, no hook, and neutral tasks.
- Held-out test: new paraphrases AND entirely withheld topics, with evidence-based
  references, multiple political framings, false-premise controls, and uncertainty.
- Calibration fitting: separate from the held-out decision test. Neither set
  may enter direction extraction, dose search, checkpoint selection, or RL training.

CLI exact-prompt hashes prevent literal reuse. They do not detect paraphrase or
topic leakage; split preparation and manual review are still required.
`data/smoke.jsonl` is engineering-only, not any of these scientific splits.

## Measurements

- Directness, refusal, deflection, and factual distortion are separate labels.
- Score atomic factual claims against cited evidence; explicitly allow disputed
  or unknown quantities. Do not equate political framing with correctness.
- Test susceptibility to opposing false premises and changes of question language.
- Track incoherence, extra verbosity, truncation, and thinking non-termination.
- Evaluate normal reasoning/code/language tasks at matched dtype, template,
  thinking setting, generation budget, and sampling parameters.
- For decisions, define the probability extraction method before measuring Brier,
  log loss, and reliability. Token likelihood and verbal confidence are different.
  Known-distribution cases permit direct probability evaluation. Separate changes
  in correctness, calibration, and abstention; report sample sizes and uncertainty.

The adjacent Intern-Decision repository contains calibration tools and a pilot
benchmark. Treat it as a possible future integration, not as the confirmed RLCD
interface or a ready-made training dataset. Do not modify that repository here.

## Before exporting a training checkpoint

Demonstrate held-out factual improvements, calibration/capability tradeoffs, and
stability across languages and thinking modes. Then compare permanent projection,
adapter training, or targeted fine-tuning against runtime steering. Each exported
candidate needs a fresh evaluation; runtime success does not imply an equivalent
weight edit. Retain the original checkpoint and full run provenance.

No numerical acceptance thresholds have been agreed yet. Do not select a final
RLCD checkpoint from this draft or from the smoke examples.
