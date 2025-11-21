from operator import itemgetter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cds_vision_tools.pytorch.classification.trainer import ClassificationTrainer


def show_example(img: torch.Tensor, label: str, predicted_label: str = None):
    """
    Displays an image along with its true label and optionally the predicted label.

    Args:
        img (Tensor): The image tensor to display.
        label (str): The true label of the image.
        predicted_label (str, optional): The predicted label of the image. Defaults to None.

    Returns:
        None

    Example:
        show_example(image_tensor, "cat", "dog")
    """
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.show()

    if predicted_label is not None:
        print("True Label:", label)
        print("Predicted Label:", predicted_label)
    else:
        print("True Label:", label)


def visualize(
    viz_class_name: str,
    trainer: ClassificationTrainer,
    val_loader: torch.utils.data.DataLoader,
):
    """
    Visualizes examples of predictions made by a classification model, focusing on a specific class.

    Args:
        viz_class_name (str): The class name to visualize.
        trainer (ClassificationTrainer): An instance of the ClassificationTrainer containing the model and training information.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

    Returns:
        None

    Example:
        visualize("cat", trainer, val_loader)
    """
    model = trainer.model

    # load the best weights
    model.load_state_dict(torch.load(trainer.best_model_name))
    model.eval()

    # define a dict to track plots
    plot_dict = {
        f"Actual_{cls_true}_Predicted_{cls_pred}": 0
        for cls_true in trainer.classes
        for cls_pred in trainer.classes
        if cls_true == viz_class_name
    }

    for X_batch, y_true_batch in iter(val_loader):
        with torch.no_grad():
            y_pred = model(X_batch)
            label = trainer.classes[torch.argmax(y_true_batch[0])]
            predicted_label = trainer.classes[torch.argmax(y_pred[0])]
            key_name = f"Actual_{label}_Predicted_{predicted_label}"
            if plot_dict.get(key_name) == 0:
                show_example(X_batch[0], label, predicted_label)
                plot_dict[key_name] = 1
                # check if all plots are done
                if sum(plot_dict.values()) >= len(plot_dict):
                    break


def save_model_errors(
    trainer: ClassificationTrainer, val_loader: torch.utils.data.DataLoader
) -> dict:
    """
    Saves images of misclassified samples from the validation dataset and logs the errors.

    Args:
        trainer (ClassificationTrainer): The trainer object containing the model and training details.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset. Must have a batch size of 1.

    Returns:
        dict: A dictionary with keys as error descriptions and values as the count of each error type.
    """
    # Get model from trainer
    model = trainer.model

    # Load the best weights
    model.load_state_dict(torch.load(trainer.best_model_name))
    model.eval()

    error_dict = {}
    save_folder = Path(f"errors/{trainer.model_file_name_prefix}/")
    save_folder.mkdir(parents=True, exist_ok=True)

    # Iterate over the validation dataset
    for X_batch, y_true_batch in val_loader:
        with torch.no_grad():
            y_pred = model(X_batch)

            if trainer.task_type == "multiclass":
                # Get predicted classes by taking the argmax of probabilities
                pred_np = y_pred.argmax(dim=1).cpu().numpy()[0]
                label_np = y_true_batch.argmax(dim=1).cpu().numpy()[0]
            elif trainer.task_type == "multilabel":
                # Get predicted classes with threshold 0.5
                pred_np = (y_pred >= 0.5).int().cpu().numpy()[0]
                label_np = y_true_batch.int().cpu().numpy()[0]

            if not (label_np == pred_np).all():
                # Generate ground truth and predicted label names
                gt_label_names = "_".join(
                    [
                        str(trainer.classes[i])
                        for i in range(len(label_np))
                        if label_np[i] == 1
                    ]
                )
                predicted_label_names = "_".join(
                    [
                        str(trainer.classes[i])
                        for i in range(len(pred_np))
                        if pred_np[i] == 1
                    ]
                )

                key_name = f"Actual_{gt_label_names}_Predicted_{predicted_label_names}"
                error_dict[key_name] = error_dict.get(key_name, 0) + 1

                # Save the misclassified image
                img_name = f"{key_name}_{error_dict[key_name]}.jpg"
                file_name = save_folder / img_name
                save_image(X_batch[0], file_name)

    # Print most frequent errors
    sorted_errors = sorted(error_dict.items(), key=itemgetter(1), reverse=True)
    for key, value in sorted_errors:
        print(f"{key} : {value}")

    return error_dict


def plot_loss_function(metrics_dict: Dict, model_file_name: str = None) -> None:
    """
    Generates and saves a plot of the loss function.

    Args:
        metrics_dict: Dictionary containing 'train_loss' and 'val_loss' lists.
        model_file_name: Name of the file to save the plot. If None, the plot is not saved.
    """
    # Extract train_loss and val_loss
    train_loss = metrics_dict["train_loss"]
    # Convert validation loss tensors to regular list
    val_loss = [val.item() for val in metrics_dict["val_loss"]]

    # Ensure the lengths match
    assert len(train_loss) == len(
        val_loss
    ), "Mismatch between train_loss and val_loss lengths."

    # Create epochs indices
    epochs = range(1, len(train_loss) + 1)

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")

    # Adding title, labels, and legend
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot if model_file_name is provided
    if model_file_name is not None:
        plt.tight_layout()
        plt.savefig(model_file_name, dpi=300)
        print(f"Saved loss plot to {model_file_name}")

    # Display the plot
    plt.show()


def generate_confusion_matrix_plot(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    classes: List[str],
    model_file_name: str = None,
    cm_normalize: str = None,
) -> np.ndarray:
    """
    Generates predictions, computes the confusion matrix, and plots it.

    Args:
        val_loader: DataLoader for the validation set.
        model: Trained model to use for predictions.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
        classes: List of class names.
        model_file_name: Path to save the confusion matrix plot. If None, the plot is not saved.
        cm_normalize: Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix will not be normalized.
    Returns:
        Computed confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_true_batch in val_loader:
            X_batch, y_true_batch = X_batch.to(device), y_true_batch.to(device)
            y_pred = model(X_batch)
            y_pred_classes = torch.argmax(y_pred, dim=1)  # Predicted class indices
            y_true_classes = torch.argmax(y_true_batch, dim=1)  # True class indices
            all_preds.extend(y_pred_classes.to(device).numpy())
            all_labels.extend(y_true_classes.to(device).numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize=cm_normalize)
    print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Set tick positions and labels
    tick_positions = np.arange(len(classes))
    plt.xticks(tick_positions, classes, rotation=45)
    plt.yticks(tick_positions, classes)

    # Annotate each cell with its value
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        # display confusion matrix in integer or percentage
        if cm_normalize == None:
            cm_annot = f"{cm[i, j]:d}"
        else:
            cm_annot = f"{cm[i, j] * 100:.2f}%"
        plt.text(
            j,
            i,
            cm_annot,
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.tight_layout()

    # Save the plot if model_file_name is provided
    if model_file_name is not None:
        save_path = f"{model_file_name}_cm.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")

    # Display the plot
    plt.show()

    return cm
