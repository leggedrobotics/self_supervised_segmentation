#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Download model checkpoints trained by Hamilton et al.
#
############################################


from os.path import join, exists
import wget
import os

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/models/models/"
model_names = []
# Optionally, uncomment to download all original models:
# model_names = ["moco_v2_800ep_pretrain.pth.tar",
#                "model_epoch_0720_iter_085000.pth",
#                "picie.pkl"]

saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_names = [
    "cityscapes_vit_base_1.ckpt",
    "cocostuff27_vit_base_5.ckpt",
    "picie_and_probes.pth",
    "potsdam_test.ckpt",
]

target_files = [join(models_dir, mn) for mn in model_names] + [join(models_dir, mn) for mn in saved_model_names]

target_urls = [model_url_root + mn for mn in model_names] + [saved_model_url_root + mn for mn in saved_model_names]

for target_file, target_url in zip(target_files, target_urls):
    if not exists(target_file):
        print("\nDownloading file from {}".format(target_url))
        wget.download(target_url, target_file)
    else:
        print("\nFound {}, skipping download".format(target_file))
