import logging
import yaml
import torch
from torch import nn
from torchvision.models import resnet18

logger = logging.getLogger(__name__)


class ClassificationModel(nn.Module):
    """
    Initiates a transfer learning model class, with a 'resnet18' model type, model weights, both edited
    for the desired output size.

    :param out_features: number of classes to classify
    :param train: Boolean value to train model or not.
    """

    def __init__(self, out_features: int, train: bool = True):

        super().__init__()

        # loading in model config file
        if train:
            self.config = self.load_config("config/config.yml")
        else:
            self.config = self.load_config("config.yml")

        # Loading Device
        if self.config["use_cuda"] and torch.cuda.is_available():
            logger.info(
                "ATTENTION: The config.yml has been set to 'use_cuda' = True. The device being used is now 'cuda'."
            )
            self.device = torch.device("cuda")
        else:
            logger.info(
                "ATTENTION: Cuda not available. The device being used is now 'cpu'."
            )
            self.device = torch.device("cpu")

        # Loading model base
        self.model_base = resnet18(weights=None)

        # setting grad to false and editing the output layer
        for param in self.model_base.parameters():
            param.requires_grad = False

        self.model_base.fc = torch.nn.Sequential(
            nn.Linear(self.model_base.fc.in_features, 512),
            nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=out_features),
        )

        # sending the model to the specified device
        self.model_base.to(self.device)

    # forward pass with the transfer learning model
    def forward(self, input_data: torch.Tensor):
        logits_results = self.model_base(input_data)
        return logits_results

    # loading the weights file in
    def load_weights(self, weights_file, device: torch.device):
        """
        Loads the weights from the weight file.

        Arguments:
            weights_file: Path to the weights .pth file.
            device: Device to load the weights onto.
        """
        weights = torch.load(weights_file, map_location=device)
        self.load_state_dict(weights)

    @staticmethod
    def load_config(config_file):
        """
        Loads the config from config file.

        Arguments:
            config_file: The path to the config file in the repository.
        """
        with open(config_file) as file:
            return yaml.safe_load(file)

