#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Segmentation demonstration with STEGO
#
# This script can be used to generate segmentations of given images with the given STEGO checkpoint
#
# Before running the script, adjust the parameters in cfg/demo_config.yaml:
# - image_dir - path to the folder with images (images from its subfolders won't be processed)
# - model_path - path to the STEGO checkpoint
# - output_root - path to the folder to save the segmentation in (segmentations will be saved in a subfolder named after experiment_name)
#
############################################


import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from stego.utils import prep_args, flexible_collate, get_transform
from stego.data import UnlabeledImageFolder, create_cityscapes_colormap
from stego.stego import Stego


torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(config_path="cfg", config_name="demo_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = os.path.join(cfg.output_root, cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "linear"), exist_ok=True)

    model = Stego.load_from_checkpoint(cfg.model_path)

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.resolution, False, "center"),
    )

    loader = DataLoader(
        dataset,
        cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )

    model.eval().cuda()
    cmap = create_cityscapes_colormap()

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = img.cuda()
            code = model.get_code(img)
            cluster_crf, linear_crf = model.postprocess(
                code=code,
                img=img,
                use_crf_cluster=cfg.run_crf,
                use_crf_linear=cfg.run_crf,
            )
            cluster_crf = cluster_crf.cpu()
            linear_crf = linear_crf.cpu()
            for j in range(img.shape[0]):
                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                Image.fromarray(cmap[linear_crf[j]].astype(np.uint8)).save(os.path.join(result_dir, "linear", new_name))
                Image.fromarray(cmap[cluster_crf[j]].astype(np.uint8)).save(
                    os.path.join(result_dir, "cluster", new_name)
                )


if __name__ == "__main__":
    prep_args()
    my_app()
