import os
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    A custom dataset class for image classification tasks, supporting both multiclass and multilabel classification.

    Args:
        images_dir_path (Path): Path to the directory containing image folders.
        classes_list (list[str]): List of class labels.
        task_name (str, optional): Type of classification task ("multiclass" or "multilabel"). Defaults to "multiclass".
        transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        annotations_df (pd.DataFrame, optional): DataFrame containing annotations for multilabel classification. Defaults to None.
        device (str, optional): Device to which the tensors should be moved ("cpu", "cuda", "mps"). Defaults to "mps".

    Attributes:
        transform (callable): Transform to apply to the images.
        annotations_df (pd.DataFrame): DataFrame containing annotations for multilabel classification.
        images_dir_path (str or Path): Path to the directory containing image folders.
        classes_list (list[str]): List of class labels.
        task_name (str): Type of classification task.
        device (str): Device to which the tensors should be moved.
        image_list (list[Path]): List of image file paths for multiclass classification.
    """

    def __init__(
        self,
        images_dir_path: Path,
        classes_list: list[str],
        task_name: str = "multiclass",
        transform: torchvision.transforms.Compose = None,
        annotations_df: pd.DataFrame = None,
        device: str = "mps",
    ):
        self.transform = transform
        self.annotations_df = annotations_df
        self.images_dir_path = images_dir_path
        self.classes_list = classes_list
        self.task_name = task_name
        self.device = device
        valid_image_extensions = [".jpg", ".jpeg", ".bmp", ".png", ".gif"]
        self.image_list = []

        for cls in self.classes_list:
            folder_path = Path(self.images_dir_path / str(cls))
            if folder_path.exists() and folder_path.is_dir():
                self.image_list += [
                    folder_path / fn
                    for fn in os.listdir(folder_path)
                    if any(fn.endswith(ext) for ext in valid_image_extensions)
                ]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        For multilabel classification, it returns the length of the annotations DataFrame.
        For multiclass classification, it returns the length of the image list.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        if self.task_name == "multilabel":
            return len(self.annotations_df)
        if self.task_name == "multiclass":
            return len(self.image_list)

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label at the specified index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image tensor and the label tensor, both moved to the specified device.
        """
        if self.task_name == "multilabel":
            image_dict = self.annotations_df.iloc[idx].to_dict()
            img_path = self.images_dir_path.parent / image_dict["image_id"]
            label = torch.Tensor([int(image_dict[cls]) for cls in self.classes_list])

        if self.task_name == "multiclass":
            img_path = self.image_list[idx]
            cls_name = Path(img_path).parent.name
            label = torch.Tensor(
                [1 if cls_name == cls else 0 for cls in self.classes_list]
            )

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img.to(self.device), label.to(self.device)
