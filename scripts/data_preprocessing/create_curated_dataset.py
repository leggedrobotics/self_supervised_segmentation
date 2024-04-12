#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Curated dataset generation script
#
# This script uses a list of samples in the input dataset to create a separate dataset containing only those samples.
# The list of samples should be saved in a text file, with each sample name in a separate line.
# Sample names need to correspond to names of image and label files in the dataset (name of an image, without file extensions).
# The samples in the new dataset are created as links to samples in the input dataset to save memory.
# Hence, the input dataset to this script should already be preprocessed.
#
# Expected input structure:
# DATA_DIR
# |-- INPUT_NAME
#     |-- imgs
#         |-- train
#         |-- val
#     |-- labels
#         |-- train
#         |-- val
#
############################################


import os
from os.path import join
import numpy as np
from tqdm import tqdm

from scripts.data_preprocessing.preprocessing_utils import *


DATA_DIR = "/data"
INPUT_NAME = "cocostuff_preprocessed"
OUTPUT_NAME = "cocostuff_curated"

TRAIN_SAMPLES_FILE = "/data/cocostuff/curated/train2017/Coco164kFull_Stuff_Coarse.txt"
VAL_SAMPLES_FILE = "/data/cocostuff/curated/val2017/Coco164kFull_Stuff_Coarse.txt"


def preprocess_samples(input_dir, output_dir, subset, input_subset, sample_file):
    print("Processing subset {}".format(subset))
    image_files = []
    label_files = []
    sample_names = []
    with open(sample_file, "r") as f:
        img_ids = [fn.rstrip() for fn in f.readlines()]
        for img_id in img_ids:
            image_files.append(join(input_dir, "imgs", input_subset, img_id + ".jpg"))
            label_files.append(join(input_dir, "labels", input_subset, img_id + ".png"))
            sample_names.append(img_id)
    for i, sample_name in tqdm(enumerate(sample_names)):
        img_path = image_files[i]
        label_path = label_files[i]
        os.link(img_path, join(output_dir, "imgs", subset, sample_name + ".jpg"))
        os.link(label_path, join(output_dir, "labels", subset, sample_name + ".png"))


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", "train", TRAIN_SAMPLES_FILE)
    preprocess_samples(input_dir, output_dir, "val", "val", VAL_SAMPLES_FILE)


if __name__ == "__main__":
    main()
