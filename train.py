import logging
import mlflow
import torch
import torch.optim as optim
import argparse
import os
import json

from torch.utils.data import Dataset
from torchmetrics import F1Score, Recall, Precision
from tqdm import tqdm
from dataset import KeyWordDetectionDataset
from model import KeywordDetectionModel

def logMetrics(epoch: int, metrics: dict, split: str):
    metricString = "split:".ljust(20) + f"[{epoch} "
    for metric in metrics:
        metricString += f"{metric} {metrics[metric]:.3f} "
        mlflow.log_metric(f"{split}/{metric}", metrics[metric])

    logging.info(metricString)


def train(
        device,
        learningRate,
        batchSize,
        epochs,
        datasetDir,
        saveMetric,
        valInterval,
        lrDecayStep,
        lrDecayGamma,
        trackingURI,
        mlflowRunName,
        modelConfig,
):
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(trackingURI)

    model = KeywordDetectionModel(modelConfig, numClasses=3)
    classWeight = [0.01849702, 0.88325471, 0.46852852]


    # model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Starting Training Session")
    logging.info(f"Number of trainable params: {pytorch_total_params}")
    trainSet = KeyWordDetectionDataset(
        os.path.join(datasetDir, "Train")
    )
    trainDataLoader = torch.utils.data.DataLoader(
        trainSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    bestScore = 0

    validationSet = KeyWordDetectionDataset(os.path.join(datasetDir, "Validation"), 16)
    validationDataLoader = torch.utils.data.DataLoader(
        validationSet,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    with mlflow.start_run(run_name=mlflowRunName) as run:
        mlflow.log_param("batchSize", batchSize)
        for key in modelConfig:
            mlflow.log_param(key, modelConfig[key])

        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(classWeight)).to(
            device
        )

        optimizer = optim.Adam(model.parameters(), lr=learningRate)
        if lrDecayStep:
            lr_stepper = torch.optim.lr_scheduler.StepLR(
                optimizer, lrDecayStep, lrDecayGamma
            )

        computeF1 = F1Score(task="multiclass", num_classes=3, average="macro")
        computeRecall = Recall(task="multiclass", num_classes=3, average="macro")
        computePrecision = Precision(task="multiclass", num_classes=3, average="macro")

        for epoch in range(epochs):
            logging.info(
                f"Starting Epoch {epoch}: Learning rate: {lr_stepper.get_last_lr() if lrDecayStep else learningRate}")
            runningTrainLoss = 0.0

            outputList = []
            targetList = []

            if True:
                model.train()
                for i, data in enumerate(tqdm(trainDataLoader)):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    with torch.autocast(device):  # fp16
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.long())
                        loss.backward()

                    # Aggregate metrics overtime to compute metrics since F1 score can't be accurately 'averaged'
                    outputList += torch.argmax(outputs, dim=1).detach().cpu()
                    targetList += labels.detach().cpu()

                    optimizer.step()

                    runningTrainLoss += loss.item()

                outputList = torch.tensor(outputList)
                targetList = torch.tensor(targetList)

                metrics = {
                    "F1": computeF1(outputList, targetList),
                    "Recall": computeRecall(outputList, targetList),
                    "Precision": computePrecision(outputList, targetList),
                    "Loss": runningTrainLoss / (len(trainSet) / batchSize),
                }

                logMetrics(epoch + 1, metrics, "Train")

            if epoch % valInterval == 0:
                model.eval()
                logging.info("Running Validation")
                outputList = []
                targetList = []
                runningValidationLoss = 0
                for i, data in enumerate(tqdm(validationDataLoader)):
                    inputs, labels = data
                    with torch.autocast(device) and torch.no_grad():
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.long())
                        runningValidationLoss += loss.item()
                    outputList += torch.argmax(outputs, dim=1).detach().cpu()
                    targetList += labels.detach().cpu()

                outputList = torch.tensor(outputList)
                targetList = torch.tensor(targetList)

                metrics = {
                    "F1": computeF1(outputList, targetList),
                    "Recall": computeRecall(outputList, targetList),
                    "Precision": computePrecision(outputList, targetList),
                    "Loss": runningValidationLoss / (len(validationSet) / batchSize),
                }

                logMetrics(epoch + 1, metrics, "Validation")

                if metrics[saveMetric] > bestScore:
                    bestScore = metrics[saveMetric]
                    logging.info(
                        f"Model {saveMetric} increased to {metrics[saveMetric]:.3f}, saving model"
                    )
                    # torch.save(model.state_dict(), "model.pth")  # Manual override to save model directly to disk
                    mlflow.pytorch.log_model(model, 'KeywordDetectionModel')
            if lrDecayStep:
                lr_stepper.step()

        logging.info("Finished Training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Script for Lunar Lander Simulation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch Size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4e-4, help="Learning rate to be used"
    )
    parser.add_argument(
        "--epochs", type=int, default=80, help="Number of epochs to train for"
    )
    parser.add_argument("--dataset_dir", type=str, help="Dataset directory")
    parser.add_argument(
        "--val",
        type=bool,
        default=False,
        help="Whether to validate or not during training",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=5,
        help="Interval at which to perform validation",
    )
    parser.add_argument(
        "--save_metrics",
        type=bool,
        default=True,
        help="Whether to save training metrics",
    )
    parser.add_argument(
        "--save_models", type=bool, default=True, help="Whether to save models"
    )
    parser.add_argument(
        "--cpu",
        type=bool,
        default=False,
        help="Uses CPU regardless of whether CUDA capable device is present",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="How often to decay learning rate",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=0.5,
        help="How much to decay learning rate by on step",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=240,
        help="Threshold for 50 episode average after which training will stop ",
    )
    parser.add_argument(
        "--save_metric",
        type=str,
        default="F1",
        help="Metric to use when saving model, one of  [F1|Accuracy|Precision]",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="The tracking URI for the mlflow server",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.getLogger().setLevel(logging.INFO)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    with open("modelConfig.json") as modelConfigFile:
        modelConfig = json.load(modelConfigFile)

    train(
        device,
        args.learning_rate,
        args.batch_size,
        args.epochs,
        args.dataset_dir,
        args.save_metric,
        args.val_interval,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.mlflow_tracking_uri,
        "keyword_detection_run",
        modelConfig,
    )
