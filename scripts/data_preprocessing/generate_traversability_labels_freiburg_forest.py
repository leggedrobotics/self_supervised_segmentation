#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Freiburg Forest traversability labels
#
# This script takes the preprocessed Freiburg Forest dataset and generates a copy with binary traversability labels
#
#
# Expected input structure after unpacking in DATA_DIR:
# DATA_DIR
# |-- OUTPUT_NAME
#     |-- imgs
#         |-- train
#         |-- val
#     |-- labels
#         |-- train
#         |-- val
#
# Output structure after processing:
# DATA_DIR
# |-- OUTPUT_NAME
#     |-- imgs
#         |-- train
#         |-- val
#     |-- labels
#         |-- train
#         |-- val
############################################


import os
from os.path import join
import numpy as np
from tqdm import tqdm
from PIL import Image

from scripts.data_preprocessing.preprocessing_utils import create_dataset_structure, preprocess_and_copy_image


DATA_DIR = "/data"
INPUT_NAME = "freiburg_forest_preprocessed"
OUTPUT_NAME = "freiburg_forest_preprocessed_trav"


FF_CMAP = np.array(
    [
        (0, 0, 0),  # Object
        (170, 170, 170),  # Road
        (0, 255, 0),  # Grass
        (102, 102, 51),  # Vegetation
        (0, 120, 255),  # Sky
        (
            0,
            60,
            0,
        ),  # Tree (separate color present in the dataset, but assigned to class Vegetation in the dataset's official readme)
    ]
)

TRAVERSABLE_IDS = [1, 2]  # Road and Grass


def preprocess_and_save_trav_label(input_name, output_name, traversable_ids):
    if os.path.isfile(output_name):
        return
    image = Image.open(input_name)
    img = np.array(image)
    label = np.zeros(img.shape)
    for id in traversable_ids:
        label = np.where(img == id, 1, label)
    image = Image.fromarray(label.astype(np.uint8))
    image.save(output_name)


def preprocess_samples(input_dir, output_dir, subset, input_subset):
    print("Processing subset {}".format(subset))
    label_names = os.listdir(join(input_dir, "labels", input_subset))
    for label_name in tqdm(label_names):
        sample_name = os.path.splitext(label_name)[0]
        img_path = join(input_dir, "imgs", input_subset, sample_name + ".jpg")
        label_path = join(input_dir, "labels", input_subset, label_name)
        preprocess_and_copy_image(img_path, join(output_dir, "imgs", subset, sample_name + ".jpg"), False)
        preprocess_and_save_trav_label(
            label_path,
            join(output_dir, "labels", subset, sample_name + ".png"),
            TRAVERSABLE_IDS,
        )


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", "train")
    preprocess_samples(input_dir, output_dir, "val", "val")


if __name__ == "__main__":
    main()
