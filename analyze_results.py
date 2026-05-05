from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

INPUT_JSONL = Path("outputs/generations.jsonl")
OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FLAT_CSV = OUTPUT_DIR / "generations_flat.csv"
CONDITION_SUMMARY_CSV = OUTPUT_DIR / "condition_summary.csv"
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
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flatten_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    flat_rows = []

    for r in records:
        base = {
            "trial_id": r.get("trial_id"),
            "replicate_id": r.get("replicate_id"),
            "scenario_id": r.get("scenario_id"),
            "provider": r.get("provider"),
            "model": r.get("model"),
            "identity_condition": r.get("identity_condition"),
            "style_condition": r.get("style_condition"),
            "email_text": r.get("email_text"),
            "success": r.get("success"),
            "error_type": r.get("error_type"),
            "error_message": r.get("error_message"),
            "latency_seconds": r.get("latency_seconds"),
            "timestamp_utc": r.get("timestamp_utc"),
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

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["success"] = df["success"].astype(bool)
    df["raise_yes"] = df["raise_recommendation"].map({"yes": 1, "no": 0})

    return df


def add_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["identity_condition"] + " / " + df["style_condition"]
    return df


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    success_df = add_condition_column(df[df["success"]].copy())

    summary = (
        success_df
        .groupby(["provider", "identity_condition", "style_condition", "condition"])
        .agg(
            n=("trial_id", "count"),
            raise_yes_rate=("raise_yes", "mean"),
            avg_raise=("recommended_raise_percent", "mean"),
            sd_raise=("recommended_raise_percent", "std"),
            avg_professionalism=("professionalism_score", "mean"),
            avg_credibility=("credibility_score", "mean"),
            avg_leadership=("leadership_score", "mean"),
            avg_persuasiveness=("persuasiveness_score", "mean"),
            avg_confidence=("confidence_score", "mean"),
        )
        .reset_index()
    )

    return summary.round(3)


def make_findings(df: pd.DataFrame, summary: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    success_df = df[df["success"]].copy()

    lines.append("INTERESTING FINDINGS")
    lines.append("====================")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Successful rows: {len(success_df)}")
    lines.append(f"Failed rows: {len(df) - len(success_df)}")

    if len(success_df) == 0:
        return lines

    lines.append("")
    lines.append("1. Highest and lowest average raise recommendations by model")
    for provider in sorted(summary["provider"].dropna().unique()):
        p = summary[summary["provider"] == provider]
        high = p.sort_values("avg_raise", ascending=False).iloc[0]
        low = p.sort_values("avg_raise", ascending=True).iloc[0]
        gap = high["avg_raise"] - low["avg_raise"]

        lines.append(
            f"- {provider}: highest = {high['condition']} ({high['avg_raise']:.2f}%), "
            f"lowest = {low['condition']} ({low['avg_raise']:.2f}%), "
            f"gap = {gap:.2f} points."
        )

    lines.append("")
    lines.append("2. Explicit gender cue effect using identical neutral writing")
    neutral = success_df[success_df["style_condition"] == "neutral"]

    for provider in sorted(neutral["provider"].dropna().unique()):
        p = neutral[neutral["provider"] == provider]
        means = p.groupby("identity_condition")["recommended_raise_percent"].mean()

        if {"woman", "man"}.issubset(means.index):
            diff = means["woman"] - means["man"]
            lines.append(
                f"- {provider}: woman label minus man label = {diff:.2f} raise-percentage points."
            )

    lines.append("")
    lines.append("3. Writing-style effect when gender is not stated")
    no_gender = success_df[success_df["identity_condition"] == "none"]

    for provider in sorted(no_gender["provider"].dropna().unique()):
        p = no_gender[no_gender["provider"] == provider]
        means = p.groupby("style_condition")["recommended_raise_percent"].mean()

        if {"feminine_coded", "masculine_coded"}.issubset(means.index):
            diff = means["feminine_coded"] - means["masculine_coded"]
            lines.append(
                f"- {provider}: feminine-coded minus masculine-coded = {diff:.2f} raise-percentage points."
            )

    lines.append("")
    lines.append("4. Cross-model comparison")
    provider_means = (
        success_df
        .groupby("provider")[NUMERIC_COLUMNS + ["raise_yes"]]
        .mean(numeric_only=True)
        .round(3)
    )
    lines.append(provider_means.to_string())

    if SCIPY_AVAILABLE:
        lines.append("")
        lines.append("5. Simple statistical checks")

        for provider in sorted(success_df["provider"].dropna().unique()):
            p = success_df[success_df["provider"] == provider]

            neutral_p = p[p["style_condition"] == "neutral"]
            w = neutral_p[neutral_p["identity_condition"] == "woman"]["recommended_raise_percent"].dropna()
            m = neutral_p[neutral_p["identity_condition"] == "man"]["recommended_raise_percent"].dropna()

            if len(w) >= 2 and len(m) >= 2:
                t, pval = stats.ttest_ind(w, m, equal_var=False)
                lines.append(
                    f"- {provider}, neutral writing woman vs man label: t={t:.3f}, p={pval:.4f}"
                )

            no_gender_p = p[p["identity_condition"] == "none"]
            f = no_gender_p[no_gender_p["style_condition"] == "feminine_coded"]["recommended_raise_percent"].dropna()
            masc = no_gender_p[no_gender_p["style_condition"] == "masculine_coded"]["recommended_raise_percent"].dropna()

            if len(f) >= 2 and len(masc) >= 2:
                t, pval = stats.ttest_ind(f, masc, equal_var=False)
                lines.append(
                    f"- {provider}, no gender feminine-coded vs masculine-coded: t={t:.3f}, p={pval:.4f}"
                )
    else:
        lines.append("")
        lines.append("5. Simple statistical checks")
        lines.append("scipy not installed, so t-tests were skipped. Install with: pip install scipy")

    return lines


def plot_avg_raise_by_condition(summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index="condition",
        columns="provider",
        values="avg_raise",
    )

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Average recommended raise percent")
    plt.title("Average Recommended Raise by Condition and Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "avg_raise_by_condition_and_model.png", dpi=200)
    plt.close()


def plot_raise_yes_rate(summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index="condition",
        columns="provider",
        values="raise_yes_rate",
    )

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("Proportion recommending raise")
    plt.ylim(0, 1)
    plt.title("Raise Recommendation Rate by Condition and Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raise_yes_rate_by_condition_and_model.png", dpi=200)
    plt.close()


def plot_score_heatmap(summary: pd.DataFrame) -> None:
    score_cols = [
        "avg_professionalism",
        "avg_credibility",
        "avg_leadership",
        "avg_persuasiveness",
        "avg_confidence",
    ]

    heat_df = summary.copy()
    heat_df["row_label"] = heat_df["provider"] + " | " + heat_df["condition"]
    heat_df = heat_df.set_index("row_label")[score_cols]

    plt.figure(figsize=(10, max(6, len(heat_df) * 0.25)))
    plt.imshow(heat_df, aspect="auto")
    plt.colorbar(label="Average score")
    plt.xticks(range(len(score_cols)), score_cols, rotation=45, ha="right")
    plt.yticks(range(len(heat_df.index)), heat_df.index)
    plt.title("Average Evaluation Scores by Condition and Model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "score_heatmap_by_condition_and_model.png", dpi=200)
    plt.close()


def plot_model_overall_means(df: pd.DataFrame) -> None:
    success_df = df[df["success"]].copy()

    overall = (
        success_df
        .groupby("provider")[NUMERIC_COLUMNS + ["raise_yes"]]
        .mean(numeric_only=True)
        .round(3)
    )

    plt.figure(figsize=(10, 6))
    overall["recommended_raise_percent"].plot(kind="bar")
    plt.ylabel("Average recommended raise percent")
    plt.title("Overall Average Recommended Raise by Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_avg_raise_by_model.png", dpi=200)
    plt.close()


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Could not find {INPUT_JSONL}")

    records = load_jsonl(INPUT_JSONL)
    df = flatten_records(records)
    df.to_csv(FLAT_CSV, index=False)

    summary = summarize_conditions(df)
    summary.to_csv(CONDITION_SUMMARY_CSV, index=False)

    findings = make_findings(df, summary)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(findings))
        f.write("\n\nCONDITION SUMMARY\n")
        f.write("=================\n")
        f.write(summary.to_string(index=False))

    plot_avg_raise_by_condition(summary)
    plot_raise_yes_rate(summary)
    plot_score_heatmap(summary)
    plot_model_overall_means(df)

    print(f"Saved flat CSV: {FLAT_CSV}")
    print(f"Saved condition summary: {CONDITION_SUMMARY_CSV}")
    print(f"Saved report: {SUMMARY_TXT}")
    print(f"Saved graphs in: {FIG_DIR}")
    print()
    print("\n".join(findings))


if __name__ == "__main__":
    main()