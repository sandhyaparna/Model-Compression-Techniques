import json
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd


def multilabel2dataframe(
    label_file: Union[str, list], classes: list[str], validation_percentage: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a multilabel annotation file into a pandas DataFrame and splits it into training and validation sets.

    Args:
        label_file (Union[str, list]): Path to the JSON file containing annotations or a list of loaded annotations.
        classes (list[str]): List of class labels.
        validation_percentage (float, optional): Percentage of data to be used for validation. Defaults to 0.2.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - Training DataFrame with the specified percentage of data.
            - Validation DataFrame with the remaining data.

    Example:
        train_df, val_df = multilabel2dataframe('annotations.json', ['class1', 'class2'], 0.2)
    """
    # if label_file is not a list, load JSON data from file.
    if type(label_file) != list:
        with open(label_file, "r") as f:
            label_file = json.load(f)

    label_columns = ["image_id"] + classes
    label_data = pd.DataFrame(columns=label_columns)

    # Loop through each annotation
    for annt_data in label_file:
        # Extract image ID from "file_upload" string
        image_id = annt_data["file_upload"].split("-")[1]

        try:
            annt = annt_data["annotations"][0]["result"][0]["value"]["choices"]
        except:
            # skip missing or wring annotations
            warnings.warn(f"Wrong annotation. {annt_data}. Skipping...")
            continue

        # Create row with image ID and label counts
        label_counts = {label: 0 for label in label_columns[1:]}
        for annotation in annt:
            label_counts[annotation] += 1
        row = [image_id] + list(label_counts.values())

        # Add the row to label_data DataFrame
        label_data.loc[len(label_data)] = row

    val_size = int(len(label_data) * validation_percentage)
    if val_size == 0:
        return label_data, pd.DataFrame(columns=label_columns)
    return label_data[:-val_size], label_data[-val_size:]


def manifest2classification(
    label_file: str,
    images_dir: Path,
    label_key: str,
    validation_percentage: float = 0.2,
):
    """
    Loads image data from AWS manifest file, splits it into train and validation sets,
    and organizes images into class folders within separate directories.

    Args:
      label_file (str): Path to the AWS manifest file.
      images_dir (Path): Directory containing the downloaded images.
      label_key (str): The key within the manifest file containing the label name.
      validation_percentage (float, optional): Percentage of validation data (0.0 - 1.0). Defaults to 0.2.

    """
    # load image data from label file
    data_dict = _load_image_data(label_file, images_dir, label_key)

    # Split data into train and validation sets
    split_dict = _split_data(data_dict, validation_percentage)

    # Organize images into class folders within train and validation directories
    _organize_images_by_split(split_dict, images_dir)


def load_multiple_multi_label_annotations(directory: Union[str, Path]) -> List[dict]:
    """
    Loads and merges annotation data from multiple multi-label annotation JSON files in the specified directory.

    Parameters:
        directory: Path to the directory containing annotation JSON files.

    Returns:
        A combined list of all annotation data.
    """
    annotations = []

    # Iterate over all JSON files in the directory
    for json_file in Path(directory).glob("*.json"):
        with json_file.open("r") as file:
            data = json.load(file)
            annotations.extend(data)  # Merging annotation data

    return annotations


def _load_image_data(
    label_file: str, images_dir: Path, label_key: str
) -> Dict[str, Dict[str, str]]:
    """
    Loads image data (labels, and paths) from a AWS manifest file, filtering for existing images.

    Args:
      label_file (str): Path to the AWS manifest file.
      Eg:
        {
          "source": "s3://datascience-ds-cpna-carve/raw-data/dirty-clean/v1/10.29.22.59_1694702794.500111.jpg",
          "datascience-carve-cleanliness-v1": 1,
          "datascience-carve-cleanliness-v1-metadata": {
            "class-name": "dirty",
            "job-name": "labeling-job/datascience-carve-cleanliness-v1",
            "confidence": 0,
            "type": "groundtruth/image-classification",
            "human-annotated": "yes",
            "creation-date": "2023-09-21T18:49:08.381977"
          }
        }
      images_dir (Path): Directory containing the downloaded images.
      label_key (str): The key within the manifest file containing the label name.

    Returns:
      Dict[str, Dict[str, str]]: A dictionary mapping image filenames to dictionaries containing label and image path.
    """

    data_dict = {}
    with open(label_file, "r") as f:
        # try:
        for line in f:
            # Parse and extract images and labels
            json_object = json.loads(line)
            if "source-ref" not in json_object:
                raise KeyError(
                    f"Missing `source-ref` key. Error processing line: {line.strip()}."
                )
            image_url = json_object.get("source-ref", "")

            if "class-name" not in json_object[label_key]:
                raise KeyError(
                    f"Missing `class-name` within `{label_key}` key. Error processing line: {line.strip()}."
                )
            label = json_object.get(label_key, {}).get("class-name", "")

            # Check for empty or missing label
            if not label:
                raise ValueError(
                    f"Empty or missing label in `{label_key}` key. Line: {line.strip()}"
                )

            image_filename = Path(image_url).name

            # Validate image existence before processing
            image_path = images_dir / image_filename
            if not image_path.is_file():
                continue  # Skip if image doesn't exist locally

            data_dict[image_filename] = {"label": label, "image_path": image_path}

    return data_dict


def _split_data(
    data: Dict[str, Any],
    validation_percentage: float,
    random_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Splits a dictionary of data into train and validation sets and returns them within a single dictionary.

    Args:
      data (Dict[str, Any]): The dictionary containing the data to split.
      validation_percentage (float): Percentage of validation data (0.0 - 1.0).
      random_seed (int, optional): Seed for the random number generator used for shuffling. Defaults to 42.
    Returns:
      Dict[str, Dict[str, Any]]: A dictionary containing two keys:
          - train: The dictionary containing the training data subset.
          - validation: The dictionary containing the validation data subset.
    """
    # Check if validation_percentage is out of range
    if not (0 <= validation_percentage <= 1):
        raise ValueError("validation_percentage must be between 0 and 1")

    # Calculate the number of data points to be included in the validation set based on the provided percentage.
    validation_size = int(len(data) * validation_percentage)
    # Convert the dictionary keys to a list for shuffling.
    data_keys = list(data.keys())
    # set random number generator with random_seed for reproducibility
    random.seed(random_seed)
    random.shuffle(data_keys)
    # Select keys for the validation set.
    validation_keys = data_keys[:validation_size]

    # Create train and validation data dictionaries based on validation_keys
    split_dict = {
        "train": {key: data[key] for key in data_keys if key not in validation_keys},
        "validation": {key: data[key] for key in validation_keys},
    }

    return split_dict


def _organize_images_by_split(
    data_dict: Dict[str, Dict[str, Any]], images_dir: Path
) -> None:
    """
    Organizes images into class folders within separate directories based on a split within the data dictionary.

    Created folder structure:
    images_dir
        - train
            - label1
                - images ...
            - label2
                - images ...
            - label3 ...
                - images ...
        - val
            - label1
                - images ...
            - label2
                - images ...
            - label3 ...
                - images ...

    Args:
      data_dict (Dict[str, Dict[str, Any]]): The dictionary containing containing 'train' and 'validation' splits.
      images_dir (Path): Directory containing the downloaded images.

    Returns:
      None
    """

    # Organize images into class folders within train and validation directories
    for split_type, data in data_dict.items():
        for image_filename, image_info in data.items():
            # Create the class directory within the split directory
            class_dir = images_dir / split_type / image_info["label"]
            class_dir.mkdir(parents=True, exist_ok=True)
            # Move the image file to the new path
            source_image_path = Path(image_info["image_path"])
            destination_path = class_dir / image_filename
            source_image_path.rename(destination_path)

    return None
