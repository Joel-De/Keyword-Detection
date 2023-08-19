import argparse
import numpy as np
import os
import random
import pickle

from pathlib import Path

from pydub import AudioSegment

CLASSES = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

MAX_SAMPLE_LENGTH = 1000  # MS
RESOLUTION = 16

BACKGROUND_CLIPS = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "running_tap.wav",
    # "pink_noise.wav", # optional files that have been excluded
    # "white_noise.wav",
]

from multiprocessing import Pool


def overlayAudio(
    backgroundAudio, foregroundClips: list, exportDir: Path, exportPath: str
):
    positionList = random.sample(
        range((len(backgroundAudio)) // (MAX_SAMPLE_LENGTH // RESOLUTION)),
        len(foregroundClips[1]),
    )
    positionList = np.array(positionList) * (
        MAX_SAMPLE_LENGTH // RESOLUTION
    )  # no audio in the 500 ms

    fileName = os.path.join(
        exportDir, exportPath, f"overlayed_audio_{foregroundClips[0]}.wav"
    )
    label = {"occurrences": [], "filename": fileName}
    position, audioClip = positionList[0], foregroundClips[1]
    position = random.randint(0, 499)
    fg = AudioSegment.from_file(audioClip[1]) + random.randint(2, 5)

    if random.randint(0, 5) == 0:
        backgroundAudio = backgroundAudio.overlay(fg, position=position) - 2
    else:
        backgroundAudio = fg

    label["occurrences"].append((audioClip[0], position, position + len(fg)))
    backgroundAudio.export(fileName)
    return label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computes the class weights for the provided dataset"
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        help="Path to export the generated dataset to, will be created if it doesn't exist",
        required=True,
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        help="Path to raw dataset to be converted",
        required=True,
    )

    parser.add_argument(
        "--clip_length",
        type=int,
        help="How long the generated clips should be, in ms",
        default=1500,
    )

    parser.add_argument(
        "--train_split",
        type=int,
        help="What percentage of the dataset should be used for training, the rest will be used for validation",
        default=80,
    )

    parser.add_argument(
        "--cpu_threads",
        type=int,
        help="How many threads to use in data processing, default is maximum available",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create directories for exported data
    os.makedirs(os.path.join(args.export_dir, "Train"), exist_ok=True)
    os.makedirs(os.path.join(args.export_dir, "Validation"), exist_ok=True)

    audioClips = []

    for audioClass in CLASSES:
        audioClips += [
            (audioClass, os.path.join(args.source_dir, audioClass, fileName))
            for fileName in [*os.listdir(os.path.join(args.source_dir, audioClass))]
        ]

    random.shuffle(audioClips)

    backgroundAudioClips = []

    for backgroundClip in BACKGROUND_CLIPS:
        backgroundAudioClips.append(
            AudioSegment.from_file(
                os.path.join(args.source_dir, "_background_noise_", backgroundClip)
            )
            - random.randint(8, 10)
        )

    print(f"Total number of audio clips: {len(audioClips)}")
    sampledAudioClips = []

    for idx, audioClip in enumerate(audioClips):
        sampledAudioClips.append((idx, audioClip))

    random.shuffle(sampledAudioClips)
    audioSamplesTrain = sampledAudioClips[
        : int(len(sampledAudioClips) * args.train_split / 100)
    ]
    audioSamplesVal = sampledAudioClips[
        int(len(sampledAudioClips) * args.train_split / 100) :
    ]

    genArgsTrain = []

    for sample in audioSamplesTrain:
        sampledBackgroundAudio = random.sample(backgroundAudioClips, 1)[0]
        start = random.randint(0, len(sampledBackgroundAudio) - args.clip_length - 1)
        genArgsTrain.append(
            (
                sampledBackgroundAudio[start : start + args.clip_length],
                sample,
                args.export_dir,
                "Train",
            )
        )

    genArgsVal = []

    for sample in audioSamplesVal:
        sampledBackgroundAudio = random.sample(backgroundAudioClips, 1)[0]
        start = random.randint(0, len(sampledBackgroundAudio) - args.clip_length - 1)
        genArgsVal.append(
            (
                sampledBackgroundAudio[start : start + args.clip_length],
                sample,
                args.export_dir,
                "Validation",
            )
        )

    threads = args.cpu_threads if args.cpu_threads else os.cpu_count()
    print(f"Using {threads} threads to generate the dataset")
    print(f"Generating {len(genArgsVal)} audio samples in Validation set")
    with Pool(threads) as p:
        occurrenceMap = p.starmap(overlayAudio, genArgsVal)
        with open(
            os.path.join(args.export_dir, "Validation", "annotations.pb"), "wb"
        ) as outputFile:
            pickle.dump(occurrenceMap, outputFile)

    print(f"Generating {len(genArgsTrain)} audio samples in Train set")
    with Pool(threads) as p:
        occurrenceMap = p.starmap(overlayAudio, genArgsTrain)
        with open(
            os.path.join(args.export_dir, "Train", "annotations.pb"), "wb"
        ) as outputFile:
            pickle.dump(occurrenceMap, outputFile)
