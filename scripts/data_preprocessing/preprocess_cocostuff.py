#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Cocostuff preprocessing script
#
# https://github.com/nightrome/cocostuff
# COCO-Stuff: Thing and Stuff Classes in Context
# H. Caesar, J. Uijlings, V. Ferrari,
# In Computer Vision and Pattern Recognition (CVPR), 2018.
#
# This preprocessing script converts fine labels of the Cocostuff dataset to 27 coarse labels of the dataset.
# Additionally, it converts grayscale, single-channel images to 3-channel images
#
# Expected input structure:
# DATA_DIR
# |-- INPUT_NAME
#     |-- images
#         |-- train2017
#         |-- val2017
#     |-- annotations
#         |-- train2017
#         |-- val2017
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

from scripts.data_preprocessing.preprocessing_utils import create_dataset_structure


DATA_DIR = "/data"
INPUT_NAME = "cocostuff"
OUTPUT_NAME = "cocostuff_preprocessed"


def cocostuff_to_27_classes(mask):
    fine_to_coarse = {
        0: 9,
        1: 11,
        2: 11,
        3: 11,
        4: 11,
        5: 11,
        6: 11,
        7: 11,
        8: 11,
        9: 8,
        10: 8,
        11: 8,
        12: 8,
        13: 8,
        14: 8,
        15: 7,
        16: 7,
        17: 7,
        18: 7,
        19: 7,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 7,
        25: 6,
        26: 6,
        27: 6,
        28: 6,
        29: 6,
        30: 6,
        31: 6,
        32: 6,
        33: 10,
        34: 10,
        35: 10,
        36: 10,
        37: 10,
        38: 10,
        39: 10,
        40: 10,
        41: 10,
        42: 10,
        43: 5,
        44: 5,
        45: 5,
        46: 5,
        47: 5,
        48: 5,
        49: 5,
        50: 5,
        51: 2,
        52: 2,
        53: 2,
        54: 2,
        55: 2,
        56: 2,
        57: 2,
        58: 2,
        59: 2,
        60: 2,
        61: 3,
        62: 3,
        63: 3,
        64: 3,
        65: 3,
        66: 3,
        67: 3,
        68: 3,
        69: 3,
        70: 3,
        71: 0,
        72: 0,
        73: 0,
        74: 0,
        75: 0,
        76: 0,
        77: 1,
        78: 1,
        79: 1,
        80: 1,
        81: 1,
        82: 1,
        83: 4,
        84: 4,
        85: 4,
        86: 4,
        87: 4,
        88: 4,
        89: 4,
        90: 4,
        91: 17,
        92: 17,
        93: 22,
        94: 20,
        95: 20,
        96: 22,
        97: 15,
        98: 25,
        99: 16,
        100: 13,
        101: 12,
        102: 12,
        103: 17,
        104: 17,
        105: 23,
        106: 15,
        107: 15,
        108: 17,
        109: 15,
        110: 21,
        111: 15,
        112: 25,
        113: 13,
        114: 13,
        115: 13,
        116: 13,
        117: 13,
        118: 22,
        119: 26,
        120: 14,
        121: 14,
        122: 15,
        123: 22,
        124: 21,
        125: 21,
        126: 24,
        127: 20,
        128: 22,
        129: 15,
        130: 17,
        131: 16,
        132: 15,
        133: 22,
        134: 24,
        135: 21,
        136: 17,
        137: 25,
        138: 16,
        139: 21,
        140: 17,
        141: 22,
        142: 16,
        143: 21,
        144: 21,
        145: 25,
        146: 21,
        147: 26,
        148: 21,
        149: 24,
        150: 20,
        151: 17,
        152: 14,
        153: 21,
        154: 26,
        155: 15,
        156: 23,
        157: 20,
        158: 21,
        159: 24,
        160: 15,
        161: 24,
        162: 22,
        163: 25,
        164: 15,
        165: 20,
        166: 17,
        167: 17,
        168: 22,
        169: 14,
        170: 18,
        171: 18,
        172: 18,
        173: 18,
        174: 18,
        175: 18,
        176: 18,
        177: 26,
        178: 26,
        179: 19,
        180: 19,
        181: 24,
    }
    new_mask = np.zeros(mask.shape)
    for class_id in fine_to_coarse:
        new_mask = np.where(mask == class_id, fine_to_coarse[class_id], new_mask)
    return new_mask.astype(np.uint8)


def preprocess_and_copy_label_cocostuff(input_name, output_name):
    if os.path.isfile(output_name):
        return
    image = Image.open(input_name)
    img = np.array(image)
    img = cocostuff_to_27_classes(img)
    image = Image.fromarray(img)
    image.save(output_name)


def preprocess_and_copy_image_cocostuff(input_name, output_name):
    if os.path.isfile(output_name):
        return
    image = Image.open(input_name).convert("RGB")
    image.save(output_name)


def preprocess_samples(input_dir, output_dir, subset, input_subset):
    print("Processing subset {}".format(subset))
    label_names = os.listdir(join(input_dir, "annotations", input_subset))
    for label_name in tqdm(label_names):
        sample_name = label_name.split(".")[0]
        img_path = join(input_dir, "images", input_subset, sample_name + ".jpg")
        label_path = join(input_dir, "annotations", input_subset, label_name)
        preprocess_and_copy_image_cocostuff(img_path, join(output_dir, "imgs", subset, sample_name + ".jpg"))
        preprocess_and_copy_label_cocostuff(label_path, join(output_dir, "labels", subset, sample_name + ".png"))


def main():
    input_dir = join(DATA_DIR, INPUT_NAME)
    output_dir = join(DATA_DIR, OUTPUT_NAME)
    create_dataset_structure(output_dir)
    preprocess_samples(input_dir, output_dir, "train", "train2017")
    preprocess_samples(input_dir, output_dir, "val", "val2017")


if __name__ == "__main__":
    main()
