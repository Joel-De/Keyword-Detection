import os
import torch
import numpy as np
import torchaudio
import logging
import pickle
import argparse
import shutil

from torch.utils.data import Dataset
from torchaudio.transforms import TimeMasking, TimeStretch, FrequencyMasking
from tqdm import tqdm

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


class KeyWordDetectionDataset(Dataset):
    mapping = {"marvin": 1, "stop": 2}  # default is 0

    def __init__(self, rootDir, cache=True):
        self.rootDir = rootDir
        self.annotations = self.loadAnnotations()

        self.cached = cache
        if cache:
            self.tmpDir = os.path.join(rootDir, "cache")
            # shutil.rmtree(self.tmpDir)
            if os.path.exists(self.tmpDir):
                logging.info(f"Cache exists for {self.rootDir}")
                return
            else:
                os.makedirs(self.tmpDir)

            logging.info(f"Caching dataset {rootDir}")
            for idx in tqdm(range(len(self.annotations))):
                waveform = self.__preprocess(self.annotations[idx]["filename"])
                with open(os.path.join(self.tmpDir, f"{idx}.pb"), "wb") as file:
                    pickle.dump(waveform, file)

    def loadAnnotations(self):
        with open(os.path.join(self.rootDir, "annotations.pb"), "rb") as inputFile:
            return pickle.load(inputFile)

    def __preprocess(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath, normalize=True)
        n_mels = 40  # this was 40
        # print(sample_rate)
        trans = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=160,
            n_fft=512,
            win_length=480,
            n_mels=n_mels,
            normalized=True
        )
        tensor = trans(waveform)
        waveform = torch.nn.functional.pad(
            tensor, (0, 121 - tensor.shape[2], 0, 0, 0, 0), "constant", 0
        )

        self.spec_aug = torch.nn.Sequential(
            TimeStretch(1.1, fixed_rate=True),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )

        # plt.imshow(trans(waveform).squeeze())
        # plt.show()
        return waveform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.cached:
            with open(os.path.join(self.tmpDir, f"{idx}.pb"), "rb") as file:
                waveform = pickle.load(file)
        else:
            waveform = self.__preprocess(self.annotations[idx]["filename"])

        annotations = self.annotations[idx]["occurrences"]

        annotationsTensor = self.mapping.get(annotations[0][0], 0)

        return waveform, annotationsTensor


def computeClassWeights(dataset: KeyWordDetectionDataset) -> list:
    res = [dataset[item][1] for item in tqdm(range(len(dataset)))]
    class_sample_count = np.unique(res, return_counts=True)[1]
    weight = 1.0 / class_sample_count
    return weight / np.linalg.norm(weight)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computes the class weights for the provided dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to dataset (parent dir of Train and Validation)",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainDataset = KeyWordDetectionDataset(os.path.join(args.dataset_dir, "Train"))
    validationDataset = KeyWordDetectionDataset(
        os.path.join(args.dataset_dir, "Validation")
    )

    trainDatasetWeights = computeClassWeights(trainDataset)
    validationDatasetWeights = computeClassWeights(validationDataset)

    print(f"The class weights for the train Dataset are: {trainDatasetWeights}")
    print(
        f"The class weights for the validation Dataset are: {validationDatasetWeights}"
    )
