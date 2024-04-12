#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Freiburg Forest preprocessing script
#
# http://deepscene.cs.uni-freiburg.de/
# Abhinav Valada, Gabriel L. Oliveira, Thomas Brox, Wolfram Burgard
# Deep Multispectral Semantic Scene Understanding of Forested Environments using Multimodal Fusion
# International Symposium on Experimental Robotics (ISER), Tokyo, Japan, 2016.
#
#
# Download Friburg Forest:
# wget http://deepscene.cs.uni-freiburg.de/static/datasets/download_freiburg_forest_annotated.sh
# bash download_freiburg_forest_annotated.sh
# tar -xzf freiburg_forest_annotated.tar.gz
# rm freiburg_forest_annotated.tar.gz*
#
#
#
# Expected input structure after unpacking in DATA_DIR:
# DATA_DIR
# |-- INPUT_NAME
#     |-- train
#         |-- rgb
#         |-- GT_color
#     |-- test
#         |-- rgb
#         |-- GT_color
#
# Output structure after preprocessing:
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

from scripts.data_preprocessing.preprocessing_utils import (
    create_dataset_structure,
    convert_rgb_label,
    preprocess_and_copy_image,
)


DATA_DIR = "/data"
INPUT_NAME = "freiburg_forest_annotated"
OUTPUT_NAME = "freiburg_forest_preprocessed"


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


def preprocess_and_copy_label_FF(input_name, output_name, cmap):
    if os.path.isfile(output_name):
        return
    image = Image.open(input_name).convert("RGB")
    img = np.array(image)
    img = convert_rgb_label(img, cmap)
    img[img == 5] = 3  # Class Tree assigned to Vegetation
    image = Image.fromarray(img)
    image.save(output_name)


def preprocess_samples(input_dir, output_dir, subset, input_subset):
    print("Processing subset {}".format(subset))
    label_names = os.listdir(join(input_dir, input_subset, "GT_color"))
    for label_name in tqdm(label_names):
        sample_name = label_name.split("_")[0]
        img_path = join(input_dir, input_subset, "rgb", sample_name + "_Clipped.jpg")
        label_path = join(input_dir, input_subset, "GT_color", label_name)
        preprocess_and_copy_image(img_path, join(output_dir, "imgs", subset, sample_name + ".jpg"), False)
        preprocess_and_copy_label_FF(
            label_path,
            join(output_dir, "labels", subset, sample_name + ".png"),
            FF_CMAP,
        )


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", "train")
    preprocess_samples(input_dir, output_dir, "val", "test")


if __name__ == "__main__":
    main()
