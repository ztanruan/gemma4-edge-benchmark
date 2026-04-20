# Analog Clock Time Reading Rubric

Use this rubric for later human or LLM review of `image_clock_time_reading`.

## Primary Check

The key correctness check is exact label match:

- reference label format: `H_MM`
- prediction must match the reference label exactly
- nearby times are still wrong

Examples:

- reference `5_40`, prediction `5_40`: correct
- reference `5_40`, prediction `5_35`: wrong
- reference `12_05`, prediction `12:05`: wrong format unless the JSON field still uses the exact label

## Required Output Contract

Pass only if:

- output is valid JSON
- keys are present as requested by the scenario prompt
- `predicted_label` is one of the allowed labels
- `confidence_band` is one of `high`, `medium`, `low`

## Evidence to Review

Check:

- the generated `response.txt`
- the run record `final_answer`
- the scenario doc in `docs/scenarios/<scenario_id>.md`
- the staged sample metadata next to the image `.json`

## Common Failure Modes

- invalid JSON
- label not in the allowed set
- human-readable time instead of exact `H_MM` label
- off-by-one 5-minute bucket
- wrong hour with correct minute
- brief_reason not grounded in visible clock-hand positions
