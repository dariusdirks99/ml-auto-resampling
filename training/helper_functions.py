import os

import numpy as np
import seaborn as sns
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from matplotlib import colormaps
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from resampling.resampling import (
    edited_nearest_neighbors,
    random_undersampling,
    random_oversampling,
    smote_sampling,
    tomelinks_sampling,
    importance_sampling,
)


def plot_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    classes: list,
    logging_folder: str = "model_logging",
    title="Confusion Matrix",
    cmap=colormaps.get_cmap(cmap="Blues"),
):
    """
    Plots the confusion matrix for a specific model true labels and predicted labels tensor.

    Args:
        y_true (np.array): The true labels, converted to an np.array
        y_pred (np.array): The model predicted labels, converted to an np.array
        classes (List): List of classes, in alphabetical order
        logging_folder (str, default='model_logging'): The path in the repository to record logging.
        title: Plot title as a string
        cmap (colormaps object): Type of coloring to use for confusion matrix
    Returns:
        A saved plot of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.xlabel("Predicted", fontdict={"fontsize": 14, "fontweight": 5})
    plt.ylabel("Actual", fontdict={"fontsize": 14, "fontweight": 5})
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    logging_path = os.path.abspath(os.path.join(current_file_dir, "..", logging_folder))
    save_path = os.path.join(logging_path, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()


def save_weights(
    model: torch.nn.Module, model_name: str, weights_folder: str = "weights"
):
    r"""
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        model_name (str): A filename for the saved model. Should include
                          either ".pth" or ".pt" as the file extension.
        weights_folder (str, default='weights'): A directory for saving the model pth file.

    Example usage:
        save_model(model=model_0,
                   save_path="C:\Users\dariu\Desktop",
                   model_name=r"05_going_modular_tingvgg_model.pth")
    """

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_file_dir, "..", f"{weights_folder}")
    model_save_path = os.path.join(weights_path, model_name)

    # Save the model state_dict()
    print(rf"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def plot_loss_curves(
    results: dict[str, list[float]], logging_folder: str = "model_logging"
):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        logging_folder (str, default='model_logging'): The folder in the repository to record logging.
    """

    # Get the loss values of the results dictionary (training and test)
    loss = []
    test_loss = []
    for i, value in enumerate(results["train_loss"]):
        loss_value = results["train_loss"][i]
        test_loss_value = results["test_loss"][i]
        loss.append(loss_value)
        test_loss.append(test_loss_value)

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(loss))

    # Setup a plot
    plt.figure(figsize=(8, 8))

    # model logging folder save paths
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    logging_path = os.path.abspath(os.path.join(current_file_dir, "..", logging_folder))
    save_path_losses = os.path.join(logging_path, "model_losses.png")
    save_path_acc = os.path.join(logging_path, "model_accuracy.png")

    # Plot loss
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(save_path_losses)
    plt.close()

    plt.figure(figsize=(8, 8))
    # Plot accuracy
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(save_path_acc)
    plt.close()


def create_summary_writer(
    model_name: str,
    extra_name: str | None = None,
    logging_folder: str = "model_logging",
):
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        model_name (str): Name of model.
        extra_name (str, optional): Anything extra to add to the directory. Defaults to None.
        logging_folder (str, default='model_logging'): The path in the repository to record logging.
    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    if extra_name:
        # Create log directory path
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        logging_path = os.path.abspath(
            os.path.join(current_file_dir, "..", logging_folder)
        )
        log_dir = os.path.join(logging_path, model_name, extra_name)
    else:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        logging_path = os.path.abspath(
            os.path.join(current_file_dir, "..", logging_folder)
        )
        log_dir = os.path.join(logging_path, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

    return SummaryWriter(log_dir=log_dir)


def load_xtrain_ytrain(data_folder_path: str, resampling: str):

    config = load_config("config.yml")

    transform = transforms.Compose([
        transforms.Resize(size=(config["preprocessing"]["resize"])),
        transforms.CenterCrop(config["preprocessing"]["crop"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["preprocessing"]["mean"],
            std=config["preprocessing"]["std"]
        ),
    ])

    dataset = ImageFolder(root=data_folder_path, transform=transform)
    # Convert dataset to numpy arrays
    x_train = []
    y_train = []
    for img, label in dataset:
        x_train.append(img.numpy())
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.array(y_train)

    # flattening
    x_train = np.stack(x_train)
    y_train = np.array(y_train)

    # Save original shape for reshaping later
    original_shape = x_train.shape[1:]  # (C, H, W)

    # Flatten to (N, C*H*W) for resampling
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    if resampling == "none":
        x_train = x_train_flat.reshape(-1, *original_shape)
        return x_train, y_train

    if resampling == "enn":
        x_train_res, y_train = edited_nearest_neighbors(x_train_flat, y_train, n_neighbors=3)
    elif resampling == "ru":
        x_train_res, y_train = random_undersampling(x_train_flat, y_train)
    elif resampling == "ro":
        x_train_res, y_train = random_oversampling(x_train_flat, y_train)
    elif resampling == "smote":
        x_train_res, y_train = smote_sampling(x_train_flat, y_train)
    elif resampling == "tl":
        x_train_res, y_train = tomelinks_sampling(x_train_flat, y_train)
    elif resampling == "is":
        x_train_res, y_train, sample_weights = importance_sampling(x_train_flat, y_train)
    else:
        raise ValueError(f"Unsupported resampling technique: {resampling}")

    x_train = x_train_res.reshape(-1, *original_shape)

    return x_train, y_train


def organize_datasets(data_folder_path: str, resampling: str = "none") -> tuple:
    config = load_config("config.yml")

    transform = transforms.Compose([
        transforms.Resize(size=(config["preprocessing"]["resize"])),
        transforms.CenterCrop(config["preprocessing"]["crop"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["preprocessing"]["mean"],
            std=config["preprocessing"]["std"]
        ),
    ])

    dataset = ImageFolder(root=data_folder_path, transform=transform)

    if resampling == "none":
        return DataLoader(dataset=dataset, batch_size=16, shuffle=True), None

    # Convert dataset to numpy arrays
    x_train, y_train = [], []
    for img, label in dataset:
        x_train.append(img.numpy())
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.array(y_train)

    # Save shape for later reshaping
    original_shape = x_train.shape[1:]  # (C, H, W)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)

    # Apply selected resampling strategy
    if resampling == "enn":
        x_train_res, y_train_res = edited_nearest_neighbors(x_train_flat, y_train, n_neighbors=3)
    elif resampling == "ru":
        x_train_res, y_train_res = random_undersampling(x_train_flat, y_train)
    elif resampling == "ro":
        x_train_res, y_train_res = random_oversampling(x_train_flat, y_train)
    elif resampling == "smote":
        x_train_res, y_train_res = smote_sampling(x_train_flat, y_train)
    elif resampling == "tl":
        x_train_res, y_train_res = tomelinks_sampling(x_train_flat, y_train)
    elif resampling == "is":
        x_train_res, y_train_res, sample_weights = importance_sampling(x_train_flat, y_train)
    else:
        raise ValueError(f"Unsupported resampling technique: {resampling}")

    # Reshape back to image tensors
    x_train = x_train_res.reshape(-1, *original_shape)
    x_tensor = torch.tensor(x_train).float()
    y_tensor = torch.tensor(y_train_res).long()

    # for importance sampling use WeightedRandomSampler
    if resampling == "is":
        weights_tensor = torch.tensor(sample_weights).float()
        sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
        dataset_resampled = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset=dataset_resampled, batch_size=16, sampler=sampler), sample_weights
    else:
        dataset_resampled = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset=dataset_resampled, batch_size=16, shuffle=True), None


def load_config(config_name: str):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_file_dir, "..", "config")
    config_file_path = os.path.join(config_path, f"{config_name}")
    with open(config_file_path) as file:
        return yaml.safe_load(file)


def set_seeds(seed: int = 689):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 689.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
