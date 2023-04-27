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
import random
from os.path import join

import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil


DATA_DIR="/media"
INPUT_NAME="RUGD"
OUTPUT_NAME="RUGD_preprocessed"

# Split according to the paper
TRAIN_SAMPLES = ["park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9",
                 "trail-10", "trail-11", "trail-12", "trail-14", "trail-15", "village"]
VAL_SAMPLES = ["park-8", "trail-5"]
TEST_SAMPLES = ["creek", "park-1", "trail-7", "trail-13"]


RUGD_CMAP = np.array([(0, 0, 0),        # void
                      (108, 64, 20),    # dirt
                      (255, 229, 204),  # sand
                      (0, 102, 0),      # grass
                      (0, 255, 0),      # tree
                      (0, 153, 153),    # pole
                      (0, 128, 255),    # water
                      (0, 0, 255),      # sky
                      (255, 255, 0),    # vehicle
                      (255, 0, 127),    # container/generic-object
                      (64, 64, 64),     # asphalt
                      (255, 128, 0),    # gravel
                      (255, 0, 0),      # building
                      (153, 76, 0),     # mulch
                      (102, 102, 0),    # rock-bed 
                      (102, 0, 0),      # log
                      (0, 255, 128),    # bicycle
                      (204, 153, 255),  # person
                      (102, 0, 204),    # fence
                      (255, 153, 204),  # bush
                      (0, 102, 102),    # sign
                      (153, 204, 255),  # rock
                      (102, 255, 255),  # bridge
                      (101, 101, 11),   # concrete
                      (114, 85, 47)])   # picnic-table


def preprocess_image(input_name, output_name, is_label=False):
    if os.path.isfile(output_name):
        return
    if is_label:
        image = Image.open(input_name).convert('RGB')
        img = np.array(image)
        for i in range(RUGD_CMAP.shape[0]):
            color = RUGD_CMAP[i]
            a = img == color
            indices = np.all(a, axis=2)
            img[indices] = i
        img = np.unique(img, axis=-1).squeeze()
        image = Image.fromarray(img)
        image.save(output_name)
    else:
        shutil.copyfile(input_name, output_name)


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, "imgs"), exist_ok=True)
    os.makedirs(join(output_dir, "imgs", "train"), exist_ok=True)
    os.makedirs(join(output_dir, "imgs", "val"), exist_ok=True)
    os.makedirs(join(output_dir, "labels"), exist_ok=True)
    os.makedirs(join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(join(output_dir, "labels", "val"), exist_ok=True)

    def preprocess_samples(subset, samples):
        print("Processing subset {}".format(subset))
        for sample in tqdm(samples):
            img_path = join(input_dir, "RUGD_frames-with-annotations", sample)
            label_path = join(input_dir, "RUGD_annotations", sample)
            img_files = os.listdir(img_path)
            label_files = os.listdir(label_path)
            for img_file in img_files:
                if img_file.endswith(('.png')):
                    preprocess_image(join(img_path, img_file), join(output_dir, "imgs", subset, img_file), False)
            print("Processing labels of sample {}".format(sample))
            for label_file in tqdm(label_files):
                if label_file.endswith(('.png')):
                    preprocess_image(join(label_path, label_file), join(output_dir, "labels", subset, label_file), True)

    preprocess_samples("train", TRAIN_SAMPLES+VAL_SAMPLES)
    preprocess_samples("val", TEST_SAMPLES)


if __name__ == "__main__":
    main()