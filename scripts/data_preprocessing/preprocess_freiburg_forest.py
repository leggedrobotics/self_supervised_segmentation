############################################
# RUGD preprocessing script
#
# RUGD: http://rugd.vision/
# Wigness, M., Eum, S., Rogers, J. G., Han, D., & Kwon, H. (2019).
# A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments. 
# International Conference on Intelligent Robots and Systems (IROS). 
#
#
#
# Download RUGD:
# wget http://rugd.vision/data/RUGD_frames-with-annotations.zip
# wget http://rugd.vision/data/RUGD_annotations.zip
#
# unzip RUGD_frames-with-annotations.zip -d RUGD
# unzip RUGD_annotations.zip -d RUGD
# rm RUGD_annotations.zip RUGD_frames-with-annotations.zip
# 
# 
# Expected input structure:
# DATA_DIR
# |-- INPUT_NAME
#     |-- RUGD_annotations
#     |-- RUGD_frames-with-annotations
#
# Output structure:
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

from scripts.data_preprocessing.preprocessing_utils import *


DATA_DIR="/media"
INPUT_NAME="Freiburg_forest"
OUTPUT_NAME="Freiburg_forest_preprocessed"


FF_CMAP = np.array([(0, 0, 0),       # Object
                    (170, 170, 170), # Trail
                    (0, 255, 0),     # Grass
                    (0, 120, 255),   # Sky
                    (102, 102, 51)]) # Vegetation


def preprocess_samples(input_dir, output_dir, subset, samples):
    print("Processing subset {}".format(subset))
    for sample in tqdm(samples):
        img_path = join(input_dir, "RUGD_frames-with-annotations", sample)
        label_path = join(input_dir, "RUGD_annotations", sample)
        img_files = os.listdir(img_path)
        label_files = os.listdir(label_path)
        for img_file in img_files:
            if img_file.endswith(('.png')):
                preprocess_and_copy_image(join(img_path, img_file),
                                          join(output_dir, "imgs", subset, img_file),
                                          False)
        print("Processing labels of sample {}".format(sample))
        for label_file in tqdm(label_files):
            if label_file.endswith(('.png')):
                preprocess_and_copy_image(join(label_path, label_file),
                                          join(output_dir, "labels", subset, label_file),
                                          True, True, RUGD_CMAP)


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", TRAIN_SAMPLES+VAL_SAMPLES)
    preprocess_samples(input_dir, output_dir, "val", TEST_SAMPLES)


if __name__ == "__main__":
    main()