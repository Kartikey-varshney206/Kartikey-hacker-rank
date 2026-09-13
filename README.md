# Buy or Wait?

An AI-assisted financial decision engine for HackerRank Orchestrate. For every purchase or payment request, it recommends whether to pay now, pay partially, use installments, wait, or decline—while protecting the user’s minimum balance.

## How it works

```text
Structured financial records ─┐
Messages and image amounts ───┼─> case dossier ─> Groq analyst ─> strict validator ─> CSV output
Payment options and FX rates ─┘                       │
                                                     └─> deterministic fallback
```

- **Groq / Qwen** interprets compact, ambiguous evidence and proposes the requested output fields.
- **Python validation** enforces output shape, allowed payment methods, exact installment schedules, payment totals, and constraints before a prediction is accepted.

Image-derived amounts are loaded from `code/extracted_image_amounts.csv`; this resolves every financial event whose input amount is blank.

## Setup

Requires Python 3.11+.

```powershell
python -m pip install -r code/requirements.txt
Copy-Item .env.example .env
```

Add your local Groq configuration:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
```

`.env` is ignored by Git. Never commit or share a real key.

## Run the sample acceptance gate

Always validate on the 25 solved samples before generating evaluation predictions.

```powershell
python code/main.py --input sample --llm --delay-seconds 20 --output llm_sample_output.csv
python code/evaluation/main.py --predictions llm_sample_output.csv
```

`--delay-seconds 20` prevents exceeding the Qwen output-token-per-minute allowance. The report identifies complete matches and every mismatched field in `evaluation/sample_report.md`.

For a quick smoke test:

```powershell
python code/main.py --input sample --llm --limit 3 --delay-seconds 60 --output llm_smoke.csv
```

## Generate evaluation predictions

Only run this after the sample score is acceptable:

```powershell
python code/main.py --input requests --output output.csv
```

The evaluator requires these exact columns, in this order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

## Project layout

```text
code/
  main.py                       Pipeline and deterministic validation
  extracted_image_amounts.csv   Resolved values for image-backed events
  requirements.txt              Runtime dependency list
  evaluation/main.py            Sample-comparison report generator
dataset/                        Challenge input data
run.py                          Sample-only launcher
```

## Safety rules enforced

- Pending credits and unrealized investment values are excluded from cash availability.
- Pending debits are reserved.
- Only methods the user accepts are eligible.
- Installment schedules must match a supplied payment option exactly.
- Partial plans require two chronological payments that sum to the requested amount.
- Only allowed flexible expenses may be changed.

## Submission checklist

- `output.csv` has one row for every row in `dataset/requests.csv`.
- Output columns and allowed values are valid.
- `evaluation/usage_report.md` is completed for the final full run.
- `code.zip`, `output.csv`, and `log.txt` are ready for submission.
