from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("outputs/duration_eval_random_5_30s_segments_all.csv")
SUMMARY_CSV = Path("outputs/duration_eval_random_5_30s_segments_all_summary.csv")
BUCKET_SUMMARY_CSV = Path("outputs/duration_eval_random_5_30s_segments_all_bucket_summary.csv")

PLOT_DIR = Path("outputs/plots_random_segments")
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
    bucket_df = pd.read_csv(BUCKET_SUMMARY_CSV)

    print("\nOverall summary:")
    print(summary_df.to_string(index=False))

    print("\nBucket summary:")
    print(bucket_df.to_string(index=False))


def print_and_save_outliers(df):
    outliers = df[
        (df["parsed_duration"] > 35)
        | (df["parsed_duration"] < 0)
        | (df["absolute_error"] > 10)
    ].copy()

    outlier_path = PLOT_DIR / "random_5_30s_outliers.csv"
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
        "Random 5–30s Segments: Actual vs Predicted Duration\n"
        f"Full range, MAE = {mae:.3f} sec"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "actual_vs_predicted_random_5_30s_full_range.png"
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
        "Random 5–30s Segments: Actual vs Predicted Duration\n"
        f"Zoomed to 0–35 sec, MAE = {mae:.3f} sec"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "actual_vs_predicted_random_5_30s_zoomed.png"
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
        "Random 5–30s Segments: Prediction Error Distribution\n"
        f"MAE = {mae:.3f} sec, Median Error = {median_error:.3f} sec"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "error_distribution_random_5_30s.png"
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
        "Random 5–30s Segments: Prediction Error Distribution\n"
        f"Errors ≤ 15 sec, MAE = {mae:.3f} sec, Median = {median_error:.3f} sec"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "error_distribution_random_5_30s_zoomed.png"
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
    plt.title("Random 5–30s Segments: Tolerance Accuracy")
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()

    output_path = PLOT_DIR / "tolerance_accuracy_random_5_30s.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_bucket_mae():
    bucket_df = pd.read_csv(BUCKET_SUMMARY_CSV)

    bucket_order = ["05-10s", "10-15s", "15-20s", "20-25s", "25-30s"]
    bucket_df["duration_bucket"] = pd.Categorical(
        bucket_df["duration_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    bucket_df = bucket_df.sort_values("duration_bucket")

    plt.figure(figsize=(8, 6))

    plt.bar(
        bucket_df["duration_bucket"].astype(str),
        bucket_df["mean_absolute_error"],
    )

    plt.xlabel("True Segment Duration Bucket")
    plt.ylabel("Mean Absolute Error (sec)")
    plt.title("Mean Absolute Error by Segment Duration Bucket")
    plt.grid(axis="y", alpha=0.3)

    for i, row in bucket_df.reset_index(drop=True).iterrows():
        plt.text(
            i,
            row["mean_absolute_error"] + 0.05,
            f"{row['mean_absolute_error']:.2f}\nn={int(row['n'])}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = PLOT_DIR / "mae_by_duration_bucket_random_5_30s.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_bucket_true_vs_predicted():
    bucket_df = pd.read_csv(BUCKET_SUMMARY_CSV)

    bucket_order = ["05-10s", "10-15s", "15-20s", "20-25s", "25-30s"]
    bucket_df["duration_bucket"] = pd.Categorical(
        bucket_df["duration_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    bucket_df = bucket_df.sort_values("duration_bucket").reset_index(drop=True)

    x = range(len(bucket_df))

    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        bucket_df["mean_true_duration"],
        marker="o",
        label="Mean true duration",
    )

    plt.plot(
        x,
        bucket_df["mean_predicted_duration"],
        marker="o",
        label="Mean predicted duration",
    )

    plt.xticks(x, bucket_df["duration_bucket"].astype(str))
    plt.xlabel("True Segment Duration Bucket")
    plt.ylabel("Duration (sec)")
    plt.title("Mean True vs Predicted Duration by Bucket")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "mean_true_vs_predicted_by_bucket_random_5_30s.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_bucket_tolerance_accuracy():
    bucket_df = pd.read_csv(BUCKET_SUMMARY_CSV)

    bucket_order = ["05-10s", "10-15s", "15-20s", "20-25s", "25-30s"]
    bucket_df["duration_bucket"] = pd.Categorical(
        bucket_df["duration_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    bucket_df = bucket_df.sort_values("duration_bucket").reset_index(drop=True)

    x = range(len(bucket_df))

    plt.figure(figsize=(9, 6))

    plt.plot(
        x,
        bucket_df["accuracy_within_1s"],
        marker="o",
        label="Within ±1s",
    )

    plt.plot(
        x,
        bucket_df["accuracy_within_2s"],
        marker="o",
        label="Within ±2s",
    )

    plt.plot(
        x,
        bucket_df["accuracy_within_5s"],
        marker="o",
        label="Within ±5s",
    )

    plt.xticks(x, bucket_df["duration_bucket"].astype(str))
    plt.ylim(0, 1)
    plt.xlabel("True Segment Duration Bucket")
    plt.ylabel("Accuracy")
    plt.title("Tolerance Accuracy by Segment Duration Bucket")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "tolerance_accuracy_by_bucket_random_5_30s.png"
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

    plot_bucket_mae()
    plot_bucket_true_vs_predicted()
    plot_bucket_tolerance_accuracy()

    print(f"\nAll plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()