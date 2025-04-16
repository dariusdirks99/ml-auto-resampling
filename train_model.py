import logging
import time

import numpy as np
import torch
from model.model import ClassificationModel
from training import helper_functions
from training.helper_functions import organize_datasets
from training.test_step import test_step
from training.train_step import train_step
import argparse

logger = logging.getLogger(__name__)
torch.set_warn_always(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with resampling.")
    parser.add_argument("--resampling-technique", type=str, default="none",
                        choices=["none", "enn", "ru", "ro", "smote", "tl", "is"],
                        help="Select a resampling technique.")
    return parser.parse_args()


def train_model(resampling_technique: str = "none") -> dict[str, list]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Returns:
        A saved .pth file, along with loss curves and confusion matrix if desired.

    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
    For example if training for epochs=2:
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]}
    """

    # Loading Config File
    config = helper_functions.load_config("config.yml")
    model_name = config["model"]["model_name"]
    epochs = config["epochs"]
    labels = config["labels"]
    gradient_clipping = config["gradient_clipping"]
    gradient_clipping_norm_value = config["gradient_norm_value"]
    learning_rate = config["learning_rate"]
    summary_writer = config["use_summary_writer"]
    plot_confusion_matrix = config["plot_confusion_matrix"]
    plot_loss_curves = config["plot_loss_curves"]
    seeds = config["training_seeds"]

    if config["use_cuda"] or torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Setting seeds for reproducibility
    helper_functions.set_seeds(seeds)

    # Initializing the Model and model parameters
    model = ClassificationModel(out_features=len(labels), train=True)

    # Initialize the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(lr=learning_rate, params=model.parameters())

    # Create empty results dictionary
    results: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # put the model on the correct device
    model.to(device)

    # pulling training and testing dataloder
    train_dataloader, sample_weights = organize_datasets(data_folder_path=r"data/train", resampling=resampling_technique)
    test_dataloader, sample_weights = organize_datasets(data_folder_path=r"data/test")

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        time.sleep(2)

        # calling train step for the data
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clipping=gradient_clipping,
            grad_norm_value=gradient_clipping_norm_value,
            sample_weights=sample_weights
        )

        # calling test step after train step
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        # Print out what's happening
        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc * 100:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc * 100:.4f}"
        )
        time.sleep(2)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Use the writer parameter to track experiments
        # See if there's a writer, if so, log to it
        if summary_writer is True:
            writer = helper_functions.create_summary_writer(
                model_name=model_name, logging_folder="model-logging"
            )
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            # Close the writer
            writer.close()

        # Plotting confusion matrix on the last epoch
        if epoch == epochs - 1:
            model.eval()

            true_labels = []
            pred_labels = []

            with torch.inference_mode():
                for batch, (X, y) in enumerate(test_dataloader):
                    X, y = torch.tensor(X).to(torch.device(device)), y.to(
                        torch.device(device)
                    )

                    # 1. Forward pass
                    test_pred_logits = model(X)

                    # 2. Argmax of results
                    test_pred_labels = test_pred_logits.argmax(dim=1)

                    # 3. Sending to CPU
                    test_pred_labels = np.array(test_pred_labels.cpu())
                    y = np.array(y.cpu())

                    true_labels.extend(y)
                    pred_labels.extend(test_pred_labels)

            # 4. Plotting confusion matrix and loss/accuracy curves'
            if plot_confusion_matrix:
                logger.info("Plotting confusion matrix and saving to logging path...")
                helper_functions.plot_confusion_matrix(
                    y_pred=pred_labels,
                    y_true=true_labels,
                    logging_folder=r"model-logging",
                    classes=config["labels"],
                )

            if plot_loss_curves:
                logger.info("Plotting loss curves and saving to logging path...")
                helper_functions.plot_loss_curves(
                    results, logging_folder=r"model-logging"
                )

    logger.info("Saving model pth file to model-file path...")
    # Save the weights of the model
    torch.save(model.state_dict(), f"model-file/{model_name}.pth")

    return results


if __name__ == "__main__":
    args = parse_args()
    train_model(resampling_technique=args.resampling_technique)
