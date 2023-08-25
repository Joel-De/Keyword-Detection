import torch
import torchaudio
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np

from dataset import KeyWordDetectionDataset
from collections import deque

# Defined by what the model was trained on
RATE = 16000

# How any sections the sliding window (of 1 second) should have,
# higher division count increases computation but increases resolution
divs = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Runs inference on an audio input device, refference "
                    "https://pytorch.org/audio/main/generated/torchaudio.io.StreamReader.html for help with"
                    "--audio_source and --audio_foramt arguments"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to saved model (downloaded from Mlflow)",
        required=True,
    )

    parser.add_argument(
        "--audio_format",
        type=str,
        help="The format specifier for stream reader, directly corresponds to '-f' argument for FFmpeg audio recordings",
        required=True,
    )

    parser.add_argument(
        "--audio_source",
        type=str,
        help="Source specifier for Stream Reader, should be a resource handler that FFmpeg can handle",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on, one of 'cuda' | 'cuda:{device}' | 'cpu'",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the scores of each class in a live heatmap, requires gui",
    )

    parser.add_argument(
        "--plot_length",
        type=int,
        default=32,
        help="How many inferences to keep track of in the plot history",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    rdr = torchaudio.io.StreamReader(
        src=f"audio={args.audio_source}", format=args.audio_format
    )

    # rdr.add_basic_audio_stream
    rdr.add_audio_stream(
        frames_per_chunk=int(RATE) // divs, filter_desc=f"aresample={RATE},afftdn"
    )

    chunks = []

    if "cuda" in args.device:  # device can be 'cuda:0'
        assert torch.cuda.is_available()

    model = torch.load(args.model_dir)
    model.eval()  # prevents gradients from being stored
    model = model.to(args.device)

    predictionHistory = deque(maxlen=args.plot_length)
    fig, ax = plt.subplots()
    plt.ion()
    while True:
        chunk = next(rdr.stream())
        chunks.append(chunk[0])
        if len(chunks) > divs:
            chunks.pop(0)
        else:
            continue

        fdata = torch.cat([chunks[idx] for idx in range(divs)], dim=0)
        audioClip = torch.swapaxes(fdata, 0, 1)[0]

        melSpecrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=RATE,
            n_mels=40,
            hop_length=160,
            n_fft=512,
            win_length=480,
            normalized=True,
        )

        waveform = melSpecrogram(audioClip).unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(args.device)

        with torch.autocast(args.device) and torch.no_grad():
            res = model(waveform)

        if args.plot:
            predictionHistory.append(res.detach().numpy()[0])
            plt.imshow(np.rot90(np.array(predictionHistory)), cmap='Blues', vmin=-10, vmax=10, interpolation='nearest')
            classes = ["Stop", "Marvin", "None"]
            ax.set_yticks(np.arange(len(classes)), labels=classes)
            plt.pause(0.1)

        res = torch.argmax(torch.squeeze(res))

        if res.item() != 0:
            print(
                dict(map(reversed, KeyWordDetectionDataset.mapping.items())).get(
                    res.item(), "None/Noise"
                )
            )
    plt.show(block=True)
