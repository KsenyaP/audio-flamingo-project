from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("outputs/duration_eval_full_audio_wordparser_all.csv")
SUMMARY_CSV = Path("outputs/duration_eval_full_audio_wordparser_all_summary.csv")

PLOT_DIR = Path("outputs/plots_full_audio")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    df = pd.read_csv(RESULTS_CSV)

    valid_df = df[
        df["true_duration"].notna()
        & df["parsed_duration"].notna()
        & (df["error"].fillna("").astype(str).str.strip() == "")
    ].copy()

    valid_df["true_duration"] = valid_df["true_duration"].astype(float)
    valid_df["parsed_duration"] = valid_df["parsed_duration"].astype(float)
    valid_df["absolute_error"] = (
        valid_df["parsed_duration"] - valid_df["true_duration"]
    ).abs()

    valid_df["within_1s"] = valid_df["absolute_error"] <= 1.0
    valid_df["within_2s"] = valid_df["absolute_error"] <= 2.0
    valid_df["within_5s"] = valid_df["absolute_error"] <= 5.0

    return valid_df


def print_summary():
    summary_df = pd.read_csv(SUMMARY_CSV)

    print("\nOverall summary:")
    print(summary_df.to_string(index=False))


def print_and_save_outliers(df):
    outliers = df[
        (df["parsed_duration"] < 0)
        | (df["absolute_error"] > 10)
    ].copy()

    outlier_path = PLOT_DIR / "full_audio_outliers.csv"
    outliers.to_csv(outlier_path, index=False)

    print(f"\nOutliers saved to: {outlier_path}")
    print(f"Number of outliers: {len(outliers)}")

    if len(outliers) > 0:
        cols = [
            "filename",
            "true_duration",
            "model_answer",
            "parsed_duration",
            "absolute_error",
        ]
        available_cols = [col for col in cols if col in outliers.columns]
        print("\nOutlier examples:")
        print(outliers[available_cols].head(30).to_string(index=False))


def plot_actual_vs_predicted_full_range(df):
    mae = df["absolute_error"].mean()

    plt.figure(figsize=(8, 7))

    plt.scatter(
        df["true_duration"],
        df["parsed_duration"],
        alpha=0.55,
        s=22,
    )

    min_val = min(df["true_duration"].min(), df["parsed_duration"].min())
    max_val = max(df["true_duration"].max(), df["parsed_duration"].max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("Actual Length (sec)")
    plt.ylabel("Predicted Length (sec)")
    plt.title(
        "Full Original Audio: Actual vs Predicted Duration\n"
        f"Full range, MAE = {mae:.3f} sec"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "actual_vs_predicted_full_audio_full_range.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_actual_vs_predicted_zoomed(df):
    mae = df["absolute_error"].mean()

    plt.figure(figsize=(8, 7))

    plt.scatter(
        df["true_duration"],
        df["parsed_duration"],
        alpha=0.55,
        s=22,
    )

    # Most TACOS files are around this range; adjust if needed.
    min_val = 0
    max_val = 35

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Actual Length (sec)")
    plt.ylabel("Predicted Length (sec)")
    plt.title(
        "Full Original Audio: Actual vs Predicted Duration\n"
        f"Zoomed to 0–35 sec, MAE = {mae:.3f} sec"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "actual_vs_predicted_full_audio_zoomed.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_error_distribution(df):
    mae = df["absolute_error"].mean()
    median_error = df["absolute_error"].median()

    plt.figure(figsize=(8, 6))

    plt.hist(df["absolute_error"], bins=30)

    plt.xlabel("Absolute Error (sec)")
    plt.ylabel("Count")
    plt.title(
        "Full Original Audio: Prediction Error Distribution\n"
        f"MAE = {mae:.3f} sec, Median Error = {median_error:.3f} sec"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "error_distribution_full_audio.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_error_distribution_zoomed(df):
    mae = df["absolute_error"].mean()
    median_error = df["absolute_error"].median()

    plt.figure(figsize=(8, 6))

    clipped_errors = df[df["absolute_error"] <= 15]["absolute_error"]
    plt.hist(clipped_errors, bins=30)

    plt.xlabel("Absolute Error (sec)")
    plt.ylabel("Count")
    plt.title(
        "Full Original Audio: Prediction Error Distribution\n"
        f"Errors ≤ 15 sec, MAE = {mae:.3f} sec, Median = {median_error:.3f} sec"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "error_distribution_full_audio_zoomed.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_tolerance_accuracy(df):
    accuracies = {
        "±1s": df["within_1s"].mean(),
        "±2s": df["within_2s"].mean(),
        "±5s": df["within_5s"].mean(),
    }

    plt.figure(figsize=(7, 5))

    labels = list(accuracies.keys())
    values = list(accuracies.values())

    plt.bar(labels, values)

    plt.ylim(0, 1)
    plt.xlabel("Tolerance")
    plt.ylabel("Accuracy")
    plt.title("Full Original Audio: Tolerance Accuracy")
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()

    output_path = PLOT_DIR / "tolerance_accuracy_full_audio.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_error_vs_true_duration(df):
    mae = df["absolute_error"].mean()

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["true_duration"],
        df["absolute_error"],
        alpha=0.55,
        s=22,
    )

    plt.xlabel("Actual Length (sec)")
    plt.ylabel("Absolute Error (sec)")
    plt.title(
        "Full Original Audio: Error vs Actual Duration\n"
        f"MAE = {mae:.3f} sec"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "error_vs_actual_duration_full_audio.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_prediction_counts(df):
    counts = (
        df["parsed_duration"]
        .round()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(9, 6))

    plt.bar(counts.index.astype(str), counts.values)

    plt.xlabel("Rounded Predicted Duration (sec)")
    plt.ylabel("Count")
    plt.title("Full Original Audio: Distribution of Rounded Predictions")
    plt.grid(axis="y", alpha=0.3)

    # Too many x labels can be unreadable, so rotate them.
    plt.xticks(rotation=90)

    plt.tight_layout()

    output_path = PLOT_DIR / "rounded_prediction_distribution_full_audio.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = load_results()

    print(f"Loaded valid examples: {len(df)}")

    print_summary()
    print_and_save_outliers(df)

    plot_actual_vs_predicted_full_range(df)
    plot_actual_vs_predicted_zoomed(df)

    plot_error_distribution(df)
    plot_error_distribution_zoomed(df)

    plot_tolerance_accuracy(df)
    plot_error_vs_true_duration(df)
    plot_prediction_counts(df)

    print(f"\nAll full-audio plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()