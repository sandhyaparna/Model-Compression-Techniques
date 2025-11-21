import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def get_mean_and_std(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        DataLoader for the dataset.

    Returns:
        Mean and standard deviation for each channel (RGB).
    """
    # Initialize variables for mean, standard deviation, and image count
    mean = torch.zeros(3)  # Mean for each channel (RGB)
    std = torch.zeros(3)  # Standard deviation for each channel
    num_images = 0

    # Iterate through batches of images in the loader
    for images, _ in loader:
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError(
                "Expected images with shape (batch_size, 3, height, width)"
            )

        batch_samples = images.size(0)  # Number of images in the batch
        num_images += batch_samples

        # Reshape images to (batch_size, channels, -1) for mean/std calculation
        images = images.view(batch_samples, images.size(1), -1)

        # Calculate mean and standard deviation across channels
        # Mean/std are calculated for each channel, and they are summed across all images in the batch
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    if num_images == 0:
        raise ValueError("No images found in DataLoader")

    # Divide by the total number of images to get the overall mean and std
    mean /= num_images
    std /= num_images

    return mean, std


def plot_label_distribution(
    images_path: Path, supported_file_types: Tuple[str] = (".png", ".jpg", ".jpeg")
) -> None:
    """
    Create bar graphs of the number of images in train and validation data folders.

    Args:
        images_path: Path to the directory containing image folders.
        supported_file_types: Tuple of supported file extensions.
    """
    label_counts: Dict[str, Dict[str, int]] = {}

    # Traverse the directory structure and collect image counts
    for root, subfolders, _ in os.walk(images_path):
        relative_path = os.path.relpath(root, images_path)
        # Skip the root directory itself (represented by ".")
        if "." not in relative_path:
            for subfolder in subfolders:
                if relative_path not in label_counts:
                    label_counts[relative_path] = {}
                subfolder_path = os.path.join(root, subfolder)
                image_count = len(
                    [
                        file
                        for file in os.listdir(subfolder_path)
                        if file.lower().endswith(supported_file_types)
                    ]
                )
                label_counts[relative_path][subfolder] = image_count

    # Plot the data
    for dataset_type, counts_dict in label_counts.items():
        class_labels, counts = zip(*counts_dict.items())
        plt.figure(figsize=(10, 5))
        plt.title(f"{dataset_type.capitalize()} data distribution", fontsize=12)
        bars = plt.bar(class_labels, counts)
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.tight_layout()

        # Annotate bars with the counts
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0, height, int(height), va="bottom"
            )

        plt.show()
