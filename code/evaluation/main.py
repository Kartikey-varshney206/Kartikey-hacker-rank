"""Compare sample predictions against the organizer-provided sample answers."""
from __future__ import annotations

import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "dataset" / "sample_requests.csv"
REPORT = ROOT / "evaluation" / "sample_report.md"
FIELDS = [
    "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
]


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["request_id"]: row for row in csv.DictReader(handle)}


def same(field: str, expected: str, actual: str) -> bool:
    if field == "amount_safe_to_pay":
        try:
            return abs(float(expected) - float(actual)) < 0.01
        except ValueError:
            return False
    return expected == actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "sample_output.csv")
    args = parser.parse_args()
    expected, predicted = load(EXPECTED), load(args.predictions)
    if set(expected) != set(predicted):
        raise ValueError("Prediction request IDs do not match the sample IDs")
    counts = {field: 0 for field in FIELDS}
    complete = 0
    details = []
    for request_id in expected:
        mismatches = []
        for field in FIELDS:
            if same(field, expected[request_id][field], predicted[request_id][field]):
                counts[field] += 1
            else:
                mismatches.append(field)
        if not mismatches:
            complete += 1
        matched = [field for field in FIELDS if field not in mismatches]
        details.append((request_id, ", ".join(matched) or "none", ", ".join(mismatches) or "none"))
    REPORT.parent.mkdir(exist_ok=True)
    lines = ["# Sample acceptance report", "", f"- Requests: {len(expected)}", f"- Exact complete matches: {complete}/{len(expected)}", "", "## Field accuracy", "", "| Field | Matches | Accuracy |", "|---|---:|---:|"]
    lines += [f"| {field} | {counts[field]}/{len(expected)} | {counts[field] / len(expected):.0%} |" for field in FIELDS]
    full_matches = [request_id for request_id, _, mismatch in details if mismatch == "none"]
    lines += ["", "## Fully matched requests", "", ", ".join(full_matches) if full_matches else "None", "", "## Per-request comparison", "", "| Request | Matched fields | Mismatched fields |", "|---|---|---|"]
    lines += [f"| {request_id} | {matched} | {mismatch} |" for request_id, matched, mismatch in details]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
