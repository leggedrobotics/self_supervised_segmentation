#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Download datasets used by Hamilton et al.
#
# In case of problems, try azcopy (see README).
#
############################################


import hydra
from omegaconf import DictConfig
import os
from os.path import join
import wget

from stego.utils import prep_args


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    dataset_names = ["potsdam", "cityscapes", "cocostuff", "potsdamraw"]
    url_base = "https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/"

    os.makedirs(pytorch_data_dir, exist_ok=True)
    for dataset_name in dataset_names:
        if (not os.path.exists(join(pytorch_data_dir, dataset_name))) or (
            not os.path.exists(join(pytorch_data_dir, dataset_name + ".zip"))
        ):
            print("\n Downloading {}".format(dataset_name))
            wget.download(
                url_base + dataset_name + ".zip",
                join(pytorch_data_dir, dataset_name + ".zip"),
            )
        else:
            print("\n Found {}, skipping download".format(dataset_name))


if __name__ == "__main__":
    prep_args()
    my_app()
