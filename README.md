# Audio Flamingo 3 Evaluation on the TACOS Dataset

This project evaluates the performance of **Audio Flamingo 3** on the TACOS audio dataset. The main focus is to test how well the model can answer questions about audio recordings under different audio granularity settings.

## Project Overview

Audio Flamingo 3 is evaluated on audio-question answering tasks using full audio recordings and, where applicable, different levels of audio segmentation or granularity.

The current implemented experiment evaluates whether Audio Flamingo 3 can estimate the duration of full, uncut audio recordings.

The model is prompted with:

```text
How long is this audio recording in seconds? Answer with only one number.
```

The predicted duration is compared with the true duration, which is computed directly from the waveform length and sampling rate.

## Dataset

This project uses the TACOS dataset.

The dataset files are expected to be stored locally in the following structure:

```text
data/
└── TACOS/
    ├── audio/
    ├── test_split.csv
    └── ...
```


## Main Files

```text
run_duration_eval.py
```

Runs the duration estimation experiment on full audio files without segmentation.

```text
af3_wrapper.py
```

Wrapper for loading and querying Audio Flamingo 3.

```text
requirements.txt
```

Python dependencies used for the project.

## Duration Evaluation

The duration evaluation script loads each audio file, sends it to Audio Flamingo 3 with a duration-related prompt, parses the model answer, and compares it to the true duration.

The true duration is computed as:

```python
true_duration = len(waveform) / sample_rate
```

The script reports:

- number of total examples
- number of valid parsed answers
- number of failed or unparsable answers
- mean absolute error
- median absolute error
- accuracy within ±1 second
- accuracy within ±2 seconds
- accuracy within ±5 seconds

## Running the Experiment

Activate the project environment:

```bash
conda activate af3
```

Run the full duration evaluation:

```bash
python run_duration_eval.py \
  --split data/TACOS/test_split.csv \
  --output outputs/duration_eval_full_audio_all.csv
```

For a small test run:

```bash
python run_duration_eval.py \
  --split data/TACOS/test_split.csv \
  --output outputs/duration_eval_full_audio_debug.csv \
  --limit 5
```

Depending on the server/GPU setup, it may be necessary to specify a GPU manually:

```bash
CUDA_VISIBLE_DEVICES=0 python run_duration_eval.py \
  --split data/TACOS/test_split.csv \
  --output outputs/duration_eval_full_audio_all.csv
```


## Output Files

The script creates:

```text
outputs/duration_eval_full_audio_all.csv
outputs/duration_eval_full_audio_all_summary.csv
```

The `outputs/` directory is ignored by Git and is not included in this repository.

## Notes on Reproducibility

The model is run with deterministic generation by default:

```text
temperature = 0.0
```

The default maximum number of generated tokens is:

```text
max_new_tokens = 20
```

The audio is loaded as a mono waveform with a sampling rate of 16 kHz.
