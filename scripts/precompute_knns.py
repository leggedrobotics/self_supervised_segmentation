#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# KNN computation for datasets used for training STEGO
#
# This script generates the KNN file for a new dataset to be used with STEGO.
# Before running the script, preprocess the dataset (including cropping).
# Adjust the path to the dataset, subsets to be processed and target resolution in cfg/knn_config.yaml
#
############################################

import os
from os.path import join
import hydra
import numpy as np
import torch.multiprocessing
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from stego.data import ContrastiveSegDataset
from stego.stego import Stego
from stego.utils import prep_args, get_transform, get_nn_file_name


def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda())[0].mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="cfg", config_name="knn_config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0)
    os.makedirs(join(cfg.data_dir, cfg.dataset_name, "nns"), exist_ok=True)

    image_sets = cfg.image_sets

    res = cfg.resolution
    n_batches = 16
    model = Stego(1).cuda()

    for image_set in image_sets:
        feature_cache_file = get_nn_file_name(cfg.data_dir, cfg.dataset_name, model.backbone_name, image_set, res)
        if not os.path.exists(feature_cache_file):
            print("{} not found, computing".format(feature_cache_file))
            dataset = ContrastiveSegDataset(
                data_dir=cfg.data_dir,
                dataset_name=cfg.dataset_name,
                image_set=image_set,
                transform=get_transform(res, False, "center"),
                target_transform=get_transform(res, True, "center"),
                model_type=model.backbone_name,
                resolution=res,
            )

            loader = DataLoader(
                dataset,
                cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )

            with torch.no_grad():
                normed_feats = get_feats(model, loader)
                all_nns = []
                step = normed_feats.shape[0] // n_batches
                print(normed_feats.shape)
                for i in tqdm(range(0, normed_feats.shape[0], step)):
                    torch.cuda.empty_cache()
                    batch_feats = normed_feats[i : i + step, :]
                    pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                    all_nns.append(torch.topk(pairwise_sims, 30)[1])
                    del pairwise_sims
                nearest_neighbors = torch.cat(all_nns, dim=0)

                np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                print("Saved NNs", model.backbone_name, cfg.dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
