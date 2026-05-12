from pathlib import Path

import pandas as pd
import librosa

from af3_wrapper import AudioFlamingoModel


DATASET_DIR = Path("data/TACOS")
AUDIO_DIR = DATASET_DIR / "audio"
TEST_SPLIT = DATASET_DIR / "test_split.csv"


def find_audio_file(filename: str) -> Path:
    filename = filename.strip()

    audio_path = AUDIO_DIR / filename
    if audio_path.exists():
        return audio_path

    matches = list(DATASET_DIR.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find audio file: {filename}")


def main():
    test_df = pd.read_csv(TEST_SPLIT)

    print("Test split columns:")
    print(test_df.columns.tolist())

    first_row = test_df.iloc[0]
    print("\nFirst test row:")
    print(first_row)

    # Most likely column name. If this fails, print columns above and adjust.
    audio_id = str(first_row["file_id"]) if "file_id" in test_df.columns else str(first_row.iloc[0])

    audio_path = find_audio_file(audio_id)

    print("\nUsing audio file:")
    print(audio_path)

    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    prompt = "Describe the audio in one sentence."

    model = AudioFlamingoModel(
        temperature=0.2,
        max_new_tokens=80
    )

    answer = model.predict(
        waveform=waveform,
        sample_rate=sample_rate,
        prompt=prompt
    )

    print("\nPrompt:")
    print(prompt)

    print("\nModel answer:")
    print(answer)


if __name__ == "__main__":
    main()