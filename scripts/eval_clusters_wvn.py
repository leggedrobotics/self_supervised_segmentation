#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# STEGO for WVN experiment
#
# This script calculates the WVN metrics of SLIC and STEGO segmentation models.
#
# Before running the script, adjust parameters in cfg/eval_wvn_config.yaml:
# - data_dir and dataset_name - the input data should be proprocessed for the WVN experiment
# - model_paths and stego_n_clusters - paths to STEGO checkpoints and the number of clusters of each model
# - output_root and experiment_name - outputs will be save in the given directory in the subfolder named after experiment_name
# - optionally, adjust other parameters, e.g. to consider SLIC segmentations, pre-image STEGO clustering
#
############################################


import os
from os.path import join

# from multiprocessing import Pool
import hydra
import torch.multiprocessing
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib as plt
from fast_slic import Slic
import kornia
from pytictoc import TicToc
import time
import warnings
import numpy as np
from PIL import Image

from stego.utils import (
    prep_args,
    plot_distributions,
    unnorm,
    WVNMetrics,
    flexible_collate,
    get_transform,
)
from stego.stego import Stego
from stego.data import ContrastiveSegDataset

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")


@hydra.main(config_path="cfg", config_name="eval_wvn_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = os.path.join(cfg.output_root, cfg.experiment_name)

    plot_dir = None
    if cfg.save_plots:
        plot_dir = join(result_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.switch_backend("agg")

    if cfg.save_vis:
        os.makedirs(join(result_dir, "img"), exist_ok=True)
        os.makedirs(join(result_dir, "label"), exist_ok=True)

    models = []
    for model_path in cfg.model_paths:
        models.append(Stego.load_from_checkpoint(model_path))

    slic_models = []
    for n_clusters in cfg.slic_n_clusters:
        slic_models.append(Slic(num_components=n_clusters, compactness=cfg.slic_compactness))

    if cfg.save_vis:
        for n_clusters in cfg.stego_n_clusters:
            os.makedirs(join(result_dir, "stego_" + str(n_clusters)), exist_ok=True)
            if cfg.cluster_stego_by_image:
                os.makedirs(join(result_dir, "stego_code_" + str(n_clusters)), exist_ok=True)
        for n_clusters in cfg.slic_n_clusters:
            os.makedirs(join(result_dir, "slic_" + str(n_clusters)), exist_ok=True)

    test_dataset = ContrastiveSegDataset(
        data_dir=cfg.data_dir,
        dataset_name=cfg.dataset_name,
        image_set="val",
        transform=get_transform(cfg.resolution, False, "center"),
        target_transform=get_transform(cfg.resolution, True, "center"),
        model_type="dino",
        resolution=cfg.resolution,
    )

    test_loader = DataLoader(
        test_dataset,
        1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )

    for model in models:
        model.eval().cuda()

    model_metrics = [
        WVNMetrics("Stego_" + str(i), i, save_plots=cfg.save_plots, output_dir=plot_dir) for i in cfg.stego_n_clusters
    ]
    slic_metrics = [
        WVNMetrics("SLIC_" + str(i), i, save_plots=cfg.save_plots, output_dir=plot_dir) for i in cfg.slic_n_clusters
    ]
    if cfg.cluster_stego_by_image:
        model_cluster_metrics = [
            WVNMetrics(
                "Stego_code_" + str(i),
                i,
                save_plots=cfg.save_plots,
                output_dir=plot_dir,
            )
            for i in cfg.stego_n_clusters
        ]

    t = TicToc()
    feature_times = []
    for i, batch in enumerate(tqdm(test_loader)):
        if cfg.n_imgs is not None and i >= cfg.n_imgs:
            break
        with torch.no_grad():
            img = batch["img"].squeeze()
            label = batch["label"].squeeze()

            if cfg.save_vis:
                image = Image.fromarray((kornia.utils.tensor_to_image(unnorm(img).cpu()) * 255).astype(np.uint8))
                image.save(join(result_dir, "img", str(i) + ".png"))
                label_img = label.cpu().detach().numpy().astype(np.uint8)
                image = Image.fromarray(label_img)
                image.save(join(result_dir, "label", str(i) + ".png"))

            features = None
            code = None

            for model_index, model in enumerate(models):
                n_clusters = cfg.stego_n_clusters[model_index]
                t.tic()
                features, code = model(batch["img"].cuda())
                feature_times.append(t.tocvalue(restart=True))
                clusters = model.postprocess_cluster(code=code, img=batch["img"], use_crf=cfg.run_crf)
                time_val = t.tocvalue()
                model_metrics[model_index].update(clusters, label, features, code, time_val)
                if cfg.save_vis:
                    image = Image.fromarray((clusters.squeeze().cpu().numpy()).astype(np.uint8))
                    image.save(join(result_dir, "stego_" + str(n_clusters), str(i) + ".png"))
                if cfg.cluster_stego_by_image:
                    t.tic()
                    clusters = model.postprocess_cluster(
                        code=code,
                        img=batch["img"],
                        use_crf=cfg.run_crf,
                        image_clustering=True,
                    )
                    time_val = t.tocvalue()
                    model_cluster_metrics[model_index].update(clusters, label, features, code, time_val)
                    if cfg.save_vis:
                        image = Image.fromarray((clusters.squeeze().cpu().numpy()).astype(np.uint8))
                        image.save(
                            join(
                                result_dir,
                                "stego_code_" + str(n_clusters),
                                str(i) + ".png",
                            )
                        )

            for model_index, model in enumerate(slic_models):
                img_np = kornia.utils.tensor_to_image(unnorm(img).cpu())
                t.tic()
                clusters = model.iterate(np.uint8(np.ascontiguousarray(img_np) * 255))
                time_val = t.tocvalue()
                slic_metrics[model_index].update(torch.from_numpy(clusters), label.cpu(), features, code, time_val)
                if cfg.save_vis:
                    n_clusters = cfg.slic_n_clusters[model_index]
                    image = Image.fromarray((clusters).astype(np.uint8))
                    image.save(join(result_dir, "slic_" + str(n_clusters), str(i) + ".png"))

    feature_times_np = np.array(feature_times)
    print("Feature extraction time:  Mean: {} Var: {}".format(np.mean(feature_times_np), np.var(feature_times_np)))

    model_values = []
    for metric in model_metrics:
        results, values = metric.compute(print_metrics=True)
        model_values.append(values)
    print()

    model_cluster_values = []
    if cfg.cluster_stego_by_image:
        for metric in model_cluster_metrics:
            results, values = metric.compute(print_metrics=True)
            model_cluster_values.append(values)
        print()

    slic_values = []
    for metric in slic_metrics:
        results, values = metric.compute(print_metrics=True)
        slic_values.append(values)

    time_now = int(time.time())
    if cfg.save_plots and cfg.save_comparison_plots:
        for metric in ["Avg_clusters", "Feature_var", "Code_var", "Time"]:
            metric_stego_values = [values[metric] for values in model_values]
            metric_stego_names = ["Stego_" + str(i) for i in cfg.stego_n_clusters]
            plot_distributions(
                metric_stego_values,
                100,
                metric_stego_names,
                metric,
                os.path.join(
                    plot_dir,
                    "Comparison_" + metric + "_Stego_" + str(time_now) + ".png",
                ),
            )
            metric_slic_values = [values[metric] for values in slic_values]
            metric_slic_names = ["SLIC_" + str(i) for i in cfg.slic_n_clusters]
            plot_distributions(
                metric_slic_values,
                100,
                metric_slic_names,
                metric,
                os.path.join(plot_dir, "Comparison_" + metric + "_SLIC_" + str(time_now) + ".png"),
            )

            metric_stego_cluster_values = []
            metric_stego_cluster_names = []
            if cfg.cluster_stego_by_image:
                metric_stego_cluster_values = [values[metric] for values in model_cluster_values]
                metric_stego_cluster_names = ["Stego_code_" + str(i) for i in cfg.stego_n_clusters]
                plot_distributions(
                    metric_stego_cluster_values,
                    100,
                    metric_stego_cluster_names,
                    metric,
                    os.path.join(
                        plot_dir,
                        "Comparison_" + metric + "_Stego_code_" + str(time_now) + ".png",
                    ),
                )

            plot_distributions(
                metric_stego_values + metric_slic_values + metric_stego_cluster_values,
                100,
                metric_stego_names + metric_slic_names + metric_stego_cluster_names,
                metric,
                os.path.join(plot_dir, "Comparison_" + metric + "_all_" + str(time_now) + ".png"),
            )


if __name__ == "__main__":
    prep_args()
    my_app()
