#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import os
from os.path import join

import numpy as np
from PIL import Image
import shutil


def create_dataset_structure(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs"), exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs", "train"), exist_ok=True)
    os.makedirs(join(dataset_dir, "imgs", "val"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(join(dataset_dir, "labels", "val"), exist_ok=True)


def convert_rgb_label(label, cmap):
    for i in range(cmap.shape[0]):
        color = cmap[i]
        indices = np.all(label == color, axis=2)
        label[indices] = i
    return np.unique(label, axis=-1).squeeze()


def preprocess_and_copy_image(input_name, output_name, is_label=False, rgb_label=False, cmap=None):
    if os.path.isfile(output_name):
        return
    if is_label and rgb_label:
        if cmap is None:
            raise ValueError("No colormap provided to convert the RGB label")
        image = Image.open(input_name).convert("RGB")
        img = np.array(image)
        img = convert_rgb_label(img, cmap)
        image = Image.fromarray(img)
        image.save(output_name)
    else:
        shutil.copyfile(input_name, output_name)
