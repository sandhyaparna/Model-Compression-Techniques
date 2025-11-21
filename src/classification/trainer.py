import importlib
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from tqdm import tqdm

from cds_vision_tools.utils.logging.logger import LoggingConfigurator

configurator = LoggingConfigurator()


def calculate_metrics(label: torch.Tensor, prob: torch.Tensor, task_type: str):
    """
    Calculate accuracy, weighted average precision, weighted average recall, and weighted average F2 score.

    Args:
        label (torch.Tensor): Ground truth labels with shape (N, number of classes).
        prob (torch.Tensor): Predicted probabilities with shape (N, number of classes).
        task_type (str): Type of classification task, either 'multiclass' or 'multilabel'.

    Returns:
        tuple: A tuple containing:
            - accuracy (float): Accuracy of the predictions.
            - precision (float): Weighted average precision of the predictions.
            - recall (float): Weighted average recall of the predictions.
            - f2_score (float): Weighted average F2 score of the predictions.
    """
    if task_type == "multiclass":
        # Get predicted classes by taking the argmax of probabilities
        pred_np = prob.argmax(dim=1).cpu().numpy()
        label_np = label.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(label_np, pred_np)
    elif task_type == "multilabel":
        # Get predicted classes with threshold 0.5
        pred_np = (prob >= 0.5).int().cpu().numpy()
        label_np = label.cpu().numpy()
        correct_labels = (pred_np == label_np).sum(1)
        accuracy = np.mean(correct_labels / len(label_np[0]))

    # Calculate metrics
    precision = precision_score(label_np, pred_np, average="weighted")
    recall = recall_score(label_np, pred_np, average="weighted")
    f2_score = fbeta_score(label_np, pred_np, beta=2, average="weighted")

    return accuracy, precision, recall, f2_score


class ClassificationTrainer:
    """
    A class to train a classification model using PyTorch.

    Attributes:
    -----------
    classes : list[str]
        List of class names.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    train_batch_size : int
        Batch size for training.
    val_batch_size: int
        Batch size for evaluation
    model_details : list
        List containing model name and weights.
    task_type : str
        Type of classification task ('multiclass' or 'multilabel').
    loss_fun : torch.nn.Module
        Loss function for training.
    max_lr : float
        Maximum learning rate for the optimizer.
    num_epochs : int
        Number of epochs for training.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    patience : int
        Patience for early stopping.
    criterion : str
        Metric to monitor for early stopping. Must be one of "val_accy", "val_precision", "val_recall", "val_f2", "train_loss", "val_loss".
    model_file_name_prefix : str
        Prefix for the saved model file name.
    device : str
        The device the use for training.
    """

    def __init__(
        self,
        classes: list[str],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_batch_size: int,
        val_batch_size: int,
        model_details: list = ["mobilenet_v3_small", "MobileNet_V3_Small_Weights"],
        task_type: str = "multiclass",
        loss_fun: torch.nn.Module = None,
        max_lr: float = 0.0001,
        num_epochs: int = 15,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        patience: int = 3,
        criterion: str = "val_loss",
        model_file_name_prefix: str = "classification_project",
        device: str = "mps",
    ) -> None:
        """
        Initializes the ClassificationTrainer with the given hyper-parameters.
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        models_module = importlib.import_module("torchvision.models")
        model_definition = getattr(models_module, model_details[0])
        weight_definition = getattr(models_module, model_details[1])
        self.model = model_definition(weights=weight_definition.DEFAULT)
        self.device = device
        self.classes = classes
        self.task_type = task_type

        # Change the last layer of the classifier
        if hasattr(self.model, "classifier"):
            # Check if the classifier is a Sequential container
            if isinstance(self.model.classifier, nn.Sequential):
                index = len(self.model.classifier) - 1
                num_ftrs = self.model.classifier[index].in_features
                self.model.classifier[index] = nn.Linear(
                    num_ftrs, len(self.classes), bias=True
                )
            # If the classifier is a single Linear layer
            elif isinstance(self.model.classifier, nn.Linear):
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = nn.Linear(
                    num_ftrs, len(self.classes), bias=True
                )
        elif hasattr(self.model, "fc"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.classes), bias=True)
        elif hasattr(self.model, "heads"):
            num_ftrs = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(num_ftrs, len(self.classes), bias=True)
        else:
            raise Exception(
                "ERROR! Model does not contain a final linear layer with name 'classifier' or 'fc' or 'heads'. Please choose one of the models from the provided list!"
            )

        self.model.to(self.device)

        # Set the loss function
        if loss_fun is not None:
            self.loss_fun = loss_fun
        elif task_type == "multilabel":
            self.loss_fun = nn.BCEWithLogitsLoss()
        elif task_type == "multiclass":
            self.loss_fun = nn.CrossEntropyLoss()

        self.max_lr = max_lr
        self.weight_decay = 1e-4
        self.grad_clip = 0.1
        self.NUM_EPOCHS = num_epochs
        self.threshold = 0.5

        # Set the optimizer and scheduler
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.max_lr, weight_decay=self.weight_decay
            )
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr,
                epochs=self.NUM_EPOCHS,
                steps_per_epoch=len(train_loader),
            )
        else:
            self.optimizer = optimizer
            self.sched = scheduler

        self.val_metrics_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_accy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f2": [],
        }

        # Set early stopping parameters
        self.patience = patience
        self.criterion = criterion
        self.pos_criterion = ["val_accy", "val_precision", "val_recall", "val_f2"]
        self.neg_criterion = ["train_loss", "val_loss"]

        # Set model file parameters
        self.model_file_name_prefix = model_file_name_prefix
        self.best_model_name = None

    def get_lr(self, optimizer):
        """
        Get the current learning rate from the optimizer.
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _get_best_score(self):
        """
        Get the initial best score based on the criterion.
        """
        if self.criterion in self.pos_criterion:
            return -9999.0
        else:
            return 9999.0

    def _has_improved(self, cur_score, best_score):
        """
        Check if the current score has improved compared to the best score.
        """
        if (self.criterion in self.pos_criterion) and (cur_score >= best_score):
            return True
        elif (self.criterion in self.neg_criterion) and (cur_score <= best_score):
            return True
        else:
            return False

    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        losses = []
        lrs = []
        i = 0
        best_score = self._get_best_score()

        for epoch in range(self.NUM_EPOCHS):
            self.model.train()

            for index, batch_data in enumerate(tqdm(iter(self.train_loader))):
                X_batch, y_true_batch = batch_data[0], batch_data[1]
                # # Forward pass
                y_pred = self.model(X_batch)
                # Calculate loss
                loss = self.loss_fun(y_pred, y_true_batch)
                # Backpropagation
                loss.backward()
                # Clip gradient
                if self.grad_clip:
                    nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                # Update parameters
                self.optimizer.step()
                # Set gradients to zero
                self.optimizer.zero_grad()
                # Record & update learning rate
                lrs.append(self.get_lr(self.optimizer))
                self.sched.step()
                # Store loss
                losses.append(float(loss.data))
                i += 1

            # Print loss
            epoch_loss = sum(losses[epoch * len(self.train_loader) :]) / (
                len(self.train_loader) * self.train_batch_size
            )
            configurator.logger.info(f"Epoch {epoch}, Loss: {epoch_loss}")

            # Validation metrics
            self.model.eval()
            val_loss = 0
            epoch_val_label = []
            epoch_val_preds = []

            # Inference speed calculation
            start_timer = time.time()

            for X_batch, y_true_batch in iter(self.val_loader):
                with torch.no_grad():
                    y_pred = self.model(X_batch)
                    val_loss += self.loss_fun(y_pred, y_true_batch)
                    if self.task_type == "multilabel":
                        y_pred = nn.Sigmoid()(y_pred)
                    # Append the current batch to the lists
                    epoch_val_label.append(y_true_batch)
                    epoch_val_preds.append(y_pred)

            duration = time.time() - start_timer

            # Concatenate all batches to form the final tensors
            epoch_val_label = torch.cat(epoch_val_label, dim=0)
            epoch_val_preds = torch.cat(epoch_val_preds, dim=0)

            # Get epoch metrics
            accuracy, precision, recall, f2_score = calculate_metrics(
                epoch_val_label, epoch_val_preds, self.task_type
            )
            val_loss = val_loss / (len(self.val_loader) * self.val_batch_size)

            # Update metrics dictionary
            metric_names = [
                "train_loss",
                "val_loss",
                "val_accy",
                "val_precision",
                "val_recall",
                "val_f2",
            ]
            metric_values = [
                epoch_loss,
                val_loss,
                accuracy,
                precision,
                recall,
                f2_score,
            ]
            for m_i in range(len(metric_names)):
                self.val_metrics_dict[metric_names[m_i]].append(metric_values[m_i])
            configurator.logger.info(
                f"Epoch {epoch}, val_loss: {val_loss}, val_accy: {accuracy}"
            )
            configurator.logger.info(
                f"Epoch {epoch}, val_precision: {precision}, val_recall: {recall}"
            )
            configurator.logger.info(f"Epoch {epoch}, val_f2: {f2_score}")
            configurator.logger.info(
                f"Inference fps non-torchserve: {(len(self.val_loader)*self.val_batch_size) / duration}"
            )
            configurator.logger.info(f" ")

            # Save best model
            if self._has_improved(
                self.val_metrics_dict[self.criterion][-1], best_score
            ):
                best_score = self.val_metrics_dict[self.criterion][-1]
                self.best_model_name = f"{self.model_file_name_prefix}.pt"
                torch.save(self.model.state_dict(), self.best_model_name)
            else:
                if len(self.val_metrics_dict[self.criterion]) > self.patience:
                    p_count = 0  # Placeholder for early stopping counter
                    for metric in self.val_metrics_dict[self.criterion][
                        -self.patience :
                    ]:
                        if not self._has_improved(metric, best_score):
                            p_count += 1
                    if p_count >= self.patience:
                        configurator.logger.info("Early Stopping!")
                        break

        return self.val_metrics_dict


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for imbalanced classification.

    Args:
        alpha: Weighting factor alpha in the focal loss formula. Default is 3.
        gamma: Focusing parameter gamma in the focal loss formula. Default is 3.
        reduction: Reduction method to use. Defaults to 'mean'.
    """

    def __init__(self, alpha: float = 3.0, gamma: float = 3.0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the focal loss between predicted and target labels.

        Args:
            inputs: Predicted logits from the model.
            targets: Ground truth labels.

        Returns:
            Focal loss value.
        """
        # Compute cross-entropy loss without reduction
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        # Compute the probability of correct classification
        pt = torch.exp(-ce_loss)
        # Apply focal loss formula
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none":
            return focal_loss
        else:
            raise ValueError(f"Reduction '{self.reduction}' is not supported")
