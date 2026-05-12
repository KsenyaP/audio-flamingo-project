# no cutting here

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from af3_wrapper import AudioFlamingoModel


DATASET_DIR = Path("data/TACOS")
AUDIO_DIR = DATASET_DIR / "audio"


def find_audio_file(filename: str) -> Path:
    """
    Finds an audio file inside data/TACOS/audio.
    """
    filename = str(filename).strip()

    direct_path = AUDIO_DIR / filename
    if direct_path.exists():
        return direct_path

    matches = list(AUDIO_DIR.rglob(filename))
    if matches:
        return matches[0]

    matches = list(DATASET_DIR.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find audio file: {filename}")


def load_audio(audio_path: Path, target_sr: int = 16000):
    """
    Loads full audio as mono waveform with 16 kHz sampling rate.
    """
    waveform, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
    return waveform, sample_rate


def parse_duration_seconds(answer: str):
    """
    Parses a duration in seconds from the model answer.

    Examples:
        "12" -> 12.0
        "12 seconds" -> 12.0
        "about 12.5 seconds" -> 12.5
        "0:12" -> 12.0
        "1:05" -> 65.0
        "ten" -> 10.0
        "twenty" -> 20.0
        "twentyfour" -> 24.0
        "twenty four" -> 24.0
        "twenty-four" -> 24.0
    """
    if answer is None:
        return None

    text = str(answer).strip().lower()

    # Remove common punctuation and normalize spaces.
    cleaned_text = text
    cleaned_text = cleaned_text.replace(",", " ")
    cleaned_text = cleaned_text.replace(".", " ")
    cleaned_text = cleaned_text.replace("seconds", "")
    cleaned_text = cleaned_text.replace("second", "")
    cleaned_text = cleaned_text.strip()
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "twentyone": 21,
        "twenty one": 21,
        "twenty-one": 21,
        "twentytwo": 22,
        "twenty two": 22,
        "twenty-two": 22,
        "twentythree": 23,
        "twenty three": 23,
        "twenty-three": 23,
        "twentyfour": 24,
        "twenty four": 24,
        "twenty-four": 24,
        "twentyfive": 25,
        "twenty five": 25,
        "twenty-five": 25,
        "twentysix": 26,
        "twenty six": 26,
        "twenty-six": 26,
        "twentyseven": 27,
        "twenty seven": 27,
        "twenty-seven": 27,
        "twentyeight": 28,
        "twenty eight": 28,
        "twenty-eight": 28,
        "twentynine": 29,
        "twenty nine": 29,
        "twenty-nine": 29,
        "thirty": 30,
        "thirtyone": 31,
        "thirty one": 31,
        "thirty-one": 31,
        "thirtytwo": 32,
        "thirty two": 32,
        "thirty-two": 32,
        "thirtythree": 33,
        "thirty three": 33,
        "thirty-three": 33,
        "thirtyfour": 34,
        "thirty four": 34,
        "thirty-four": 34,
        "thirtyfive": 35,
        "thirty five": 35,
        "thirty-five": 35,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
    }

    # First try exact number-word answers.
    if cleaned_text in number_words:
        return float(number_words[cleaned_text])

    # Also handle answers like "about twenty seconds".
    for phrase, value in sorted(number_words.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned_text):
            return float(value)

    # Match mm:ss or hh:mm:ss format.
    time_match = re.search(r"\b(?:(\d+):)?(\d{1,2}):(\d{2})\b", text)
    if time_match:
        hours_or_none = time_match.group(1)
        minutes = int(time_match.group(2))
        seconds = int(time_match.group(3))

        if hours_or_none is not None:
            hours = int(hours_or_none)
            return hours * 3600 + minutes * 60 + seconds

        return minutes * 60 + seconds

    # Match normal decimal/integer number.
    number_match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if number_match:
        return float(number_match.group(0))

    return None


def classify_error(abs_error: float):
    """
    Converts absolute error into useful tolerance flags.
    """
    if abs_error is None or np.isnan(abs_error):
        return {
            "within_1s": False,
            "within_2s": False,
            "within_5s": False,
        }

    return {
        "within_1s": abs_error <= 1.0,
        "within_2s": abs_error <= 2.0,
        "within_5s": abs_error <= 5.0,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="data/TACOS/test_split.csv",
        help="CSV split file with a column named 'filename'."
    )

    parser.add_argument(
        "--output",
        default="outputs/duration_eval_full_audio_wordparser.csv",
        help="Output CSV file."
    )

    parser.add_argument(
        "--prompt",
        default="How long is this audio recording in seconds? Answer with only one number.",
        help="Prompt asked to Audio Flamingo."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of files to process, useful for testing."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Use 0.0 for deterministic answers."
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of generated tokens."
    )

    args = parser.parse_args()

    split_path = Path(args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_path)

    if "filename" not in df.columns:
        raise ValueError(
            f"Expected a column named 'filename'. Found columns: {df.columns.tolist()}"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} files from {split_path}")
    print("Mode: full original audio files, no cutting")
    print("Parser: digits + number words")
    print("Loading Audio Flamingo model once...")

    model = AudioFlamingoModel(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )

    results = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        filename = str(row["filename"]).strip()

        result = {
            "index": index,
            "filename": filename,
            "audio_path": "",
            "true_duration": None,
            "prompt": args.prompt,
            "model_answer": "",
            "parsed_duration": None,
            "absolute_error": None,
            "within_1s": False,
            "within_2s": False,
            "within_5s": False,
            "error": "",
        }

        try:
            audio_path = find_audio_file(filename)
            waveform, sample_rate = load_audio(audio_path, target_sr=16000)

            true_duration = len(waveform) / sample_rate

            answer = model.predict(
                waveform=waveform,
                sample_rate=sample_rate,
                prompt=args.prompt,
            )

            parsed_duration = parse_duration_seconds(answer)

            if parsed_duration is not None:
                absolute_error = abs(parsed_duration - true_duration)
                tolerance_flags = classify_error(absolute_error)
            else:
                absolute_error = None
                tolerance_flags = classify_error(None)

            result.update({
                "audio_path": str(audio_path),
                "true_duration": true_duration,
                "model_answer": answer,
                "parsed_duration": parsed_duration,
                "absolute_error": absolute_error,
                "within_1s": tolerance_flags["within_1s"],
                "within_2s": tolerance_flags["within_2s"],
                "within_5s": tolerance_flags["within_5s"],
            })

        except Exception as e:
            result["error"] = str(e)

        results.append(result)

        # Save after every example, so progress is not lost if the job stops.
        pd.DataFrame(results).to_csv(output_path, index=False)

    results_df = pd.DataFrame(results)

    valid_df = results_df[
        results_df["parsed_duration"].notna()
        & results_df["true_duration"].notna()
        & (results_df["error"] == "")
    ]

    print("\nFinished.")
    print(f"Saved results to: {output_path}")

    print("\nSummary:")
    print(f"Total examples: {len(results_df)}")
    print(f"Valid parsed answers: {len(valid_df)}")
    print(f"Failed / unparsable: {len(results_df) - len(valid_df)}")

    if len(valid_df) > 0:
        summary = {
            "total_examples": len(results_df),
            "valid_parsed_answers": len(valid_df),
            "failed_or_unparsable": len(results_df) - len(valid_df),
            "mean_absolute_error": valid_df["absolute_error"].mean(),
            "median_absolute_error": valid_df["absolute_error"].median(),
            "accuracy_within_1s": valid_df["within_1s"].mean(),
            "accuracy_within_2s": valid_df["within_2s"].mean(),
            "accuracy_within_5s": valid_df["within_5s"].mean(),
        }

        print(f"Mean absolute error: {summary['mean_absolute_error']:.3f} seconds")
        print(f"Median absolute error: {summary['median_absolute_error']:.3f} seconds")
        print(f"Accuracy within ±1s: {summary['accuracy_within_1s']:.3f}")
        print(f"Accuracy within ±2s: {summary['accuracy_within_2s']:.3f}")
        print(f"Accuracy within ±5s: {summary['accuracy_within_5s']:.3f}")

        summary_path = output_path.with_name(output_path.stem + "_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()