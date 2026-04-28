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
            "provider": r.get("provider", "openai"),
            "model": r.get("model"),
            "identity_condition": r.get("identity_condition"),
            "style_condition": r.get("style_condition"),
            "success": r.get("success"),
            "error": r.get("error"),
            "latency_seconds": r.get("latency_seconds"),
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


def condition_label(row: pd.Series) -> str:
    return f"{row['identity_condition']} / {row['style_condition']}"


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    success_df = df[df["success"]].copy()

    summary = (
        success_df
        .groupby(["identity_condition", "style_condition"])
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

    summary["condition"] = summary.apply(condition_label, axis=1)
    return summary.round(3)


def make_findings(df: pd.DataFrame, summary: pd.DataFrame) -> List[str]:
    lines = []
    success_df = df[df["success"]].copy()

    lines.append("INTERESTING FINDINGS")
    lines.append("====================")
    lines.append(f"Total successful model runs: {len(success_df)}")

    if len(success_df) == 0:
        lines.append("No successful runs to analyze.")
        return lines

    # Highest and lowest recommended raise
    highest_raise = summary.sort_values("avg_raise", ascending=False).iloc[0]
    lowest_raise = summary.sort_values("avg_raise", ascending=True).iloc[0]
    raise_gap = highest_raise["avg_raise"] - lowest_raise["avg_raise"]

    lines.append("")
    lines.append("1. Largest raise recommendation gap")
    lines.append(
        f"The highest average raise was for {highest_raise['condition']} "
        f"at {highest_raise['avg_raise']:.2f}%."
    )
    lines.append(
        f"The lowest average raise was for {lowest_raise['condition']} "
        f"at {lowest_raise['avg_raise']:.2f}%."
    )
    lines.append(f"Gap: {raise_gap:.2f} percentage points.")

    # Explicit gender contrast, same neutral writing
    neutral = success_df[success_df["style_condition"] == "neutral"]
    if not neutral.empty:
        gender_means = (
            neutral.groupby("identity_condition")[NUMERIC_COLUMNS + ["raise_yes"]]
            .mean(numeric_only=True)
            .round(3)
        )

        lines.append("")
        lines.append("2. Explicit gender cue effect with identical neutral writing")
        lines.append(gender_means.to_string())

        if {"woman", "man"}.issubset(gender_means.index):
            diff = (
                gender_means.loc["woman", "recommended_raise_percent"]
                - gender_means.loc["man", "recommended_raise_percent"]
            )
            lines.append(
                f"Female-label minus male-label recommended raise difference: {diff:.2f} points."
            )

    # Style contrast, no gender stated
    no_gender = success_df[success_df["identity_condition"] == "none"]
    if not no_gender.empty:
        style_means = (
            no_gender.groupby("style_condition")[NUMERIC_COLUMNS + ["raise_yes"]]
            .mean(numeric_only=True)
            .round(3)
        )

        lines.append("")
        lines.append("3. Writing-style effect when gender is not stated")
        lines.append(style_means.to_string())

        if {"feminine_coded", "masculine_coded"}.issubset(style_means.index):
            diff = (
                style_means.loc["feminine_coded", "recommended_raise_percent"]
                - style_means.loc["masculine_coded", "recommended_raise_percent"]
            )
            lines.append(
                f"Feminine-coded minus masculine-coded recommended raise difference: {diff:.2f} points."
            )

    # Most variable condition
    most_variable = summary.sort_values("sd_raise", ascending=False).iloc[0]
    lines.append("")
    lines.append("4. Most unstable condition across the 10 runs")
    lines.append(
        f"{most_variable['condition']} had the highest standard deviation in recommended raise "
        f"({most_variable['sd_raise']:.2f})."
    )

    # Optional t-tests
    lines.append("")
    lines.append("5. Simple statistical checks")
    if not SCIPY_AVAILABLE:
        lines.append("scipy is not installed, so t-tests were skipped. Install with: pip install scipy")
    else:
        # woman vs man, neutral writing
        w = neutral[neutral["identity_condition"] == "woman"]["recommended_raise_percent"].dropna()
        m = neutral[neutral["identity_condition"] == "man"]["recommended_raise_percent"].dropna()

        if len(w) >= 2 and len(m) >= 2:
            t, p = stats.ttest_ind(w, m, equal_var=False)
            lines.append(
                f"Neutral writing, woman label vs man label, raise percent: t={t:.3f}, p={p:.4f}"
            )

        # feminine vs masculine, no gender
        f = no_gender[no_gender["style_condition"] == "feminine_coded"]["recommended_raise_percent"].dropna()
        masc = no_gender[no_gender["style_condition"] == "masculine_coded"]["recommended_raise_percent"].dropna()

        if len(f) >= 2 and len(masc) >= 2:
            t, p = stats.ttest_ind(f, masc, equal_var=False)
            lines.append(
                f"No gender stated, feminine-coded vs masculine-coded, raise percent: t={t:.3f}, p={p:.4f}"
            )

    return lines


def plot_avg_raise(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("avg_raise", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["condition"], plot_df["avg_raise"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average recommended raise percent")
    plt.title("Average Recommended Raise by Condition")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "avg_raise_by_condition.png", dpi=200)
    plt.close()


def plot_score_heatmap(summary: pd.DataFrame) -> None:
    score_cols = [
        "avg_professionalism",
        "avg_credibility",
        "avg_leadership",
        "avg_persuasiveness",
        "avg_confidence",
    ]

    heat_df = summary.set_index("condition")[score_cols]

    plt.figure(figsize=(10, 6))
    plt.imshow(heat_df, aspect="auto")
    plt.colorbar(label="Average score")
    plt.xticks(range(len(score_cols)), score_cols, rotation=45, ha="right")
    plt.yticks(range(len(heat_df.index)), heat_df.index)
    plt.title("Average Evaluation Scores by Condition")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "score_heatmap_by_condition.png", dpi=200)
    plt.close()


def plot_replicate_variability(df: pd.DataFrame) -> None:
    success_df = df[df["success"]].copy()
    success_df["condition"] = (
        success_df["identity_condition"] + " / " + success_df["style_condition"]
    )

    conditions = sorted(success_df["condition"].unique())
    data = [
        success_df[success_df["condition"] == c]["recommended_raise_percent"].dropna()
        for c in conditions
    ]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=conditions)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Recommended raise percent")
    plt.title("Variation Across 10 Runs by Condition")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raise_variability_boxplot.png", dpi=200)
    plt.close()


def plot_raise_yes_rate(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("raise_yes_rate", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["condition"], plot_df["raise_yes_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Proportion of runs recommending a raise")
    plt.ylim(0, 1)
    plt.title("Raise Recommendation Rate by Condition")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raise_yes_rate_by_condition.png", dpi=200)
    plt.close()


def main() -> None:
    records = load_jsonl(INPUT_JSONL)
    df = flatten_records(records)

    df.to_csv(FLAT_CSV, index=False)

    summary = summarize_conditions(df)
    summary.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)

    findings = make_findings(df, summary)

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(findings))
        f.write("\n\nCONDITION SUMMARY\n")
        f.write("=================\n")
        f.write(summary.to_string(index=False))

    plot_avg_raise(summary)
    plot_score_heatmap(summary)
    plot_replicate_variability(df)
    plot_raise_yes_rate(summary)

    print(f"Saved flat CSV: {FLAT_CSV}")
    print(f"Saved condition summary: {OUTPUT_DIR / 'condition_summary.csv'}")
    print(f"Saved written findings: {SUMMARY_TXT}")
    print(f"Saved graphs in: {FIG_DIR}")

    print("\n".join(findings))


if __name__ == "__main__":
    main()