import argparse
import librosa

from af3_wrapper import AudioFlamingoModel


def load_audio(audio_path: str, target_sr: int = 16000):
    waveform, sample_rate = librosa.load(audio_path, sr=target_sr, mono=True)
    return waveform, sample_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file, e.g. audio/example.wav")
    parser.add_argument(
        "--prompt",
        default="Describe the audio.",
        help="Prompt/question for Audio Flamingo."
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=80)

    args = parser.parse_args()

    print("Loading audio...")
    waveform, sample_rate = load_audio(args.audio, target_sr=16000)

    print("Instantiating AudioFlamingoModel...")
    model = AudioFlamingoModel(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )

    print("Running prediction...")
    answer = model.predict(
        waveform=waveform,
        sample_rate=sample_rate,
        prompt=args.prompt
    )

    print("\nPrompt:")
    print(args.prompt)
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()