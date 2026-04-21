from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Optional stats package
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

INPUT_JSONL = Path("outputs/generations.jsonl")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FLAT_CSV = OUTPUT_DIR / "generations_flat.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary_report.txt"


NUMERIC_COLUMNS = [
    "recommended_raise_percent",
    "professionalism_score",
    "credibility_score",
    "leadership_score",
    "persuasiveness_score",
    "confidence_score",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    flat_rows = []

    for r in records:
        base = {
            "trial_id": r.get("trial_id"),
            "scenario_id": r.get("scenario_id"),
            "provider": r.get("provider"),
            "model": r.get("model"),
            "identity_condition": r.get("identity_condition"),
            "style_condition": r.get("style_condition"),
            "email_text": r.get("email_text"),
            "success": r.get("success"),
            "error_type": r.get("error_type"),
            "error_message": r.get("error_message"),
            "timestamp_utc": r.get("timestamp_utc"),
            "latency_seconds": r.get("latency_seconds"),
            "raw_output_text": r.get("raw_output_text"),
        }

        parsed = r.get("parsed_output") or {}

        for key in [
            "raise_recommendation",
            "recommended_raise_percent",
            "professionalism_score",
            "credibility_score",
            "leadership_score",
            "persuasiveness_score",
            "confidence_score",
            "improvement_suggestions",
        ]:
            base[key] = parsed.get(key)

        flat_rows.append(base)

    df = pd.DataFrame(flat_rows)

    # Normalize types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "success" in df.columns:
        df["success"] = df["success"].astype(bool)

    if "raise_recommendation" in df.columns:
        df["raise_yes"] = df["raise_recommendation"].map({"yes": 1, "no": 0})

    return df


def write_section(lines: List[str], title: str) -> None:
    lines.append("")
    lines.append("=" * len(title))
    lines.append(title)
    lines.append("=" * len(title))


def basic_summary(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []

    write_section(lines, "Overall summary")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Successful rows: {int(df['success'].sum())}")
    lines.append(f"Failed rows: {int((~df['success']).sum())}")

    success_df = df[df["success"]].copy()
    lines.append(f"Rows used for quantitative analysis: {len(success_df)}")

    if len(success_df) == 0:
        return lines

    write_section(lines, "Success rate by provider")
    provider_counts = (
        df.groupby("provider")["success"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_total", "sum": "n_success", "mean": "success_rate"})
        .reset_index()
    )
    lines.append(provider_counts.to_string(index=False))

    write_section(lines, "Average scores by provider")
    provider_means = (
        success_df.groupby("provider")[NUMERIC_COLUMNS + ["raise_yes", "latency_seconds"]]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )
    lines.append(provider_means.to_string(index=False))

    write_section(lines, "Average scores by identity condition")
    identity_means = (
        success_df.groupby("identity_condition")[NUMERIC_COLUMNS + ["raise_yes"]]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )
    lines.append(identity_means.to_string(index=False))

    write_section(lines, "Average scores by style condition")
    style_means = (
        success_df.groupby("style_condition")[NUMERIC_COLUMNS + ["raise_yes"]]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )
    lines.append(style_means.to_string(index=False))

    write_section(lines, "Average scores by provider × identity × style")
    combo_means = (
        success_df.groupby(["provider", "identity_condition", "style_condition"])[NUMERIC_COLUMNS + ["raise_yes"]]
        .mean(numeric_only=True)
        .round(3)
        .reset_index()
    )
    lines.append(combo_means.to_string(index=False))

    return lines


def simple_contrasts(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    success_df = df[df["success"]].copy()

    if len(success_df) == 0:
        return lines

    write_section(lines, "Key contrasts")

    # Identity-only contrast on neutral style
    neutral_style = success_df[success_df["style_condition"] == "neutral"].copy()
    if not neutral_style.empty:
        lines.append("Identity contrast within neutral style:")
        contrast = (
            neutral_style.groupby(["provider", "identity_condition"])[NUMERIC_COLUMNS + ["raise_yes"]]
            .mean(numeric_only=True)
            .round(3)
            .reset_index()
        )
        lines.append(contrast.to_string(index=False))

    # Style-only contrast where gender is unspecified
    no_identity = success_df[success_df["identity_condition"] == "none"].copy()
    if not no_identity.empty:
        lines.append("")
        lines.append("Style contrast when gender is unspecified:")
        contrast = (
            no_identity.groupby(["provider", "style_condition"])[NUMERIC_COLUMNS + ["raise_yes"]]
            .mean(numeric_only=True)
            .round(3)
            .reset_index()
        )
        lines.append(contrast.to_string(index=False))

    return lines

def print_quick_tables(df: pd.DataFrame) -> None:
    success_df = df[df["success"]].copy()

    print("\n=== Preview of flattened data ===")
    print(df.head().to_string(index=False))

    print("\n=== Success counts by provider ===")
    print(
        df.groupby("provider")["success"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_total", "sum": "n_success", "mean": "success_rate"})
        .round(3)
        .to_string()
    )

    if len(success_df) > 0:
        print("\n=== Provider means ===")
        print(
            success_df.groupby("provider")[NUMERIC_COLUMNS + ["raise_yes", "latency_seconds"]]
            .mean(numeric_only=True)
            .round(3)
            .to_string()
        )


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Could not find {INPUT_JSONL}")

    records = load_jsonl(INPUT_JSONL)
    df = flatten_records(records)
    df.to_csv(FLAT_CSV, index=False)

    print(f"Saved flattened CSV to: {FLAT_CSV}")
    print_quick_tables(df)

    report_lines: List[str] = []
    report_lines.extend(basic_summary(df))
    report_lines.extend(simple_contrasts(df))

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines).strip() + "\n")

    print(f"\nSaved summary report to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()