import torch
import json

from builtins import super
from torch import nn
from torch.nn import functional as torchFunc


def padding(ksize):
    return (ksize - 1) // 2


class BroadcastedResidualModule(nn.Module):
    """
    Broadcast Residual Block, as defined in the reference paper, the keyword detection model consists of a series of these
        blocks in different configurations.
    """

    def __init__(
        self,
        inDim: int,
        outDim: int,
        stride: tuple = (1, 1),
        kernelSizeFrequency: int = 3,
        kernelSizeTime: int = 3,
    ) -> None:
        super().__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.stride = stride

        self.frequencyConvolution = []
        if self.inDim != self.outDim:
            # If the input and output dimensions don't match, transform the input to the required output dimension first
            self.frequencyConvolution.append(
                nn.Conv2d(inDim, outDim, (1, 1), bias=True)
            )
            self.frequencyConvolution.append(nn.BatchNorm2d(outDim))
            self.frequencyConvolution.append(nn.ReLU(True))
            inDim = outDim

        # Definition of frequency layers
        self.frequencyConvolution += [
            nn.Conv2d(
                inDim,
                outDim,
                (kernelSizeFrequency, 1),
                padding=(padding(kernelSizeFrequency), 0),
                groups=inDim,
                stride=stride,
            ),
            nn.SiLU(True),
            nn.BatchNorm2d(outDim),
        ]

        # Convert list of modules into a torch callable module
        self.frequencyConvolution = nn.Sequential(*self.frequencyConvolution)

        self.frequencyAveragePool = nn.AdaptiveAvgPool2d((1, None))

        # Definition of time/temporal layers
        self.timeConvolution = nn.Conv2d(
            outDim,
            outDim,
            (1, kernelSizeTime),
            padding=(0, padding(kernelSizeTime)),
            bias=False,
            groups=outDim,
        )
        self.swish = nn.SiLU(True)
        self.batchNormalization = nn.BatchNorm2d(outDim)
        self.identifyConvolution = nn.Conv2d(outDim, outDim, 1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        original = (
            x  # Keep a copy of the original input for the main residual connection
        )
        x = self.frequencyConvolution(
            x
        )  # Perform a convolution in the frequency domain
        frequencyConvOut = x  # Keep a copy of the output from the frequency convolution for the secondary residual connection
        x = self.frequencyAveragePool(x)

        x = self.timeConvolution(x)  # perform a convolution in the time domain
        x = self.swish(x)
        x = self.batchNormalization(x)

        x = self.identifyConvolution(x)
        x = (
            x + frequencyConvOut
        )  # Residual connection between frequency convolution & original input
        if not self.inDim != self.outDim and self.stride == (1, 1):
            """
            If the input dimensions and output dimensions of the block ar ethe same add a residual connection between
            the original input and the output (size's will mismatch if inDim != outDim, or if the stride in a single
            dim is not 1)
            """

            x = original + x
        return torchFunc.relu(x, True)


class KeywordDetectionModel(nn.Module):
    def __init__(self, modelConfig: dict, numClasses: int = 2):
        """
        Initializes an instance of the KeywordDetection model
        :param modelConfig: Configuration dictionary for models internal hyperparameters
        :param numClasses: The number of classes to predict for (controls output dimensions of classifier)
        """
        super().__init__()
        self.numLayers = modelConfig["numLayers"]
        self.numBlocks = len(self.numLayers)
        self.numClasses = numClasses

        self.channelSize = [
            modelConfig["channels"] * 2,
            modelConfig["channels"],
            int(modelConfig["channels"] * 1.5),
            modelConfig["channels"] * 2,
            int(modelConfig["channels"] * 2.5),
            modelConfig["channels"] * 4,
        ]
        assert len(self.channelSize) - 2 == self.numBlocks

        self.useStride = set(modelConfig["useStride"])  # which blocks to use stride for

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(1, self.channelSize[0], 7, (2, 1), 2),
            nn.BatchNorm2d(self.channelSize[0]),
            torch.nn.ReLU(),
        )

        broadcastedResidualBlocksList = []
        for idx, layerCount in enumerate(self.numLayers):
            stride = (2, 1) if idx in self.useStride else (1, 1)
            broadcastedResidualBlocksList.append(
                BroadcastedResidualModule(
                    self.channelSize[idx], self.channelSize[idx + 1], stride
                )
            )
            for layer in range(layerCount - 1):
                broadcastedResidualBlocksList.append(
                    BroadcastedResidualModule(
                        self.channelSize[idx + 1], self.channelSize[idx + 1], (1, 1)
                    )
                )

        self.feedForwardBase = torch.nn.Sequential(*broadcastedResidualBlocksList)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.channelSize[-2],
                self.channelSize[-2],
                (5, 5),
                bias=False,
                groups=self.channelSize[-2],
                padding=(0, 2),
            ),
            nn.Conv2d(self.channelSize[-2], self.channelSize[-1], 1, bias=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.channelSize[-1], self.numClasses, 1, bias=False),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.head(x)
        x = self.feedForwardBase(x)
        x = self.classifier(x)
        if __name__ == "__main__":
            # at the time of writing there's a bug with the Onnx exporter that causes the export script to fail if
            # torch.squeeze() is called, this circumvents that so we can see actual output from Netron
            return x
        x = torch.squeeze(x, dim=[2, 3])
        return x


if __name__ == "__main__":
    """
    This main function generates the .onnx files for the following modules
    1) The entire model
    2) The BroadcastedResidualModule with equal inDim & outDim parameters
    3) The BroadcastedResidualModules with unequal inDim &outDim parameters
    """

    logging.getLogger().setLevel(logging.INFO)
    with open("modelConfig.json") as modelConfigFile:
        modelConfig = json.load(modelConfigFile)

    # Generate .onnx file for the entire model
    dummyInput = torch.rand((32, 1, 40, 16 * 90))
    module = KeywordDetectionModel(modelConfig, numClasses=12)

    out = module(dummyInput)
    pytorch_total_params = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    logging.info(f"Number of trainable params: {pytorch_total_params}")
    torch.onnx.export(module, dummyInput, "Model.onnx")

    # Generate .onnx file for Module with equal inDim & outDim
    dummyInput = torch.rand((1, 8, 256, 256))
    module = BroadcastedResidualModule(inDim=8, outDim=8)

    out = module(dummyInput)
    torch.onnx.export(module, dummyInput, "Module_equal_dims.onnx")

    # Generate .onnx file for Module with unequal inDim & outDim
    dummyInput = torch.rand((1, 8, 256, 256))
    module = BroadcastedResidualModule(inDim=8, outDim=16)

    out = module(dummyInput)
    torch.onnx.export(module, dummyInput, "Module_unequal_dims.onnx")
