#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Plotting script for STEGO experiments
#
# Two types of plots are available:
# - Correspondence plot - an interactive plot visualizing cosine similarities
#   between all features in the image and the selected query feature
# - Precision-recall curves - a given STEGO checkpoint can be evaluated
#   on input data in predicting label co-occurrence with feature similarities
#
# Before running, adjust the parameters in cfg/plot_config.yaml
#
############################################


from os.path import join
import hydra

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import precision_recall_curve, average_precision_score

from stego.stego import Stego
from stego.data import ContrastiveSegDataset
from stego.utils import (
    prep_args,
    get_transform,
    remove_axes,
    sample,
    tensor_correlation,
    prep_for_plot,
    load_image_to_tensor,
    norm
)


class Plotter:
    """
    This class collects methods used for plot generation.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.model_path is not None:
            self.stego = Stego.load_from_checkpoint(cfg.model_path).cuda()
        else:
            self.stego = Stego(1).cuda()

    def reset_axes(self, axes):
        axes[0].clear()
        remove_axes(axes)
        axes[0].set_title("Image A and Query Point", fontsize=20)
        axes[1].set_title("Feature Cosine Similarity", fontsize=20)
        axes[2].set_title("Image B", fontsize=20)

    def get_heatmaps(self, img, img_pos, query_points, zero_mean=True, zero_clamp=True):
        """
        Runs STEGO on the given pair of images (img, img_pos)
        Generates a 2D heatmap of cosine similarities between STEGO's backbone features
        """

        feats1, _ = self.stego.forward(img.cuda())
        feats2, _ = self.stego.forward(img_pos.cuda())

        sfeats1 = sample(feats1, query_points)

        attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
        if zero_mean:
            attn_intra -= attn_intra.mean([3, 4], keepdims=True)
        if zero_clamp:
            attn_intra = attn_intra.clamp(0).squeeze(0)
        else:
            attn_intra = attn_intra.squeeze(0)

        attn_inter = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats2, dim=1))
        if zero_mean:
            attn_inter -= attn_inter.mean([3, 4], keepdims=True)
        if zero_clamp:
            attn_inter = attn_inter.clamp(0).squeeze(0)
        else:
            attn_inter = attn_inter.squeeze(0)

        heatmap_intra = (
            F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
        )
        heatmap_inter = (
            F.interpolate(attn_inter, img_pos.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
        )

        return heatmap_intra, heatmap_inter

    def plot_figure(self, img_a, img_b, query_point, axes, fig):
        """
        Plots a single visualization in the interactive correspondence figure.
        """
        _, heatmap_correspondence = self.get_heatmaps(
            img_a,
            img_b,
            query_point,
            zero_mean=self.cfg.zero_mean,
            zero_clamp=self.cfg.zero_clamp,
        )
        point = ((query_point[0, 0, 0] + 1) / 2 * self.cfg.display_resolution).cpu()
        self.reset_axes(axes)
        axes[0].imshow(prep_for_plot(img_a[0], rescale=False))
        axes[2].imshow(prep_for_plot(img_b[0], rescale=False))
        axes[0].scatter(point[0], point[1], color=(1, 0, 0), marker="x", s=500, linewidths=5)

        img_b_bw = prep_for_plot(img_b[0], rescale=False) * 0.8
        img_b_bw = np.ones_like(img_b_bw) * np.expand_dims(
            np.dot(np.array(img_b_bw)[..., :3], [0.2989, 0.5870, 0.1140]), -1
        )
        axes[1].imshow(img_b_bw)
        im1 = None
        if self.cfg.zero_clamp:
            im1 = axes[1].imshow(
                heatmap_correspondence[0],
                alpha=0.5,
                cmap=self.cfg.cmap,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            im1 = axes[1].imshow(
                heatmap_correspondence[0],
                alpha=0.5,
                cmap=self.cfg.cmap,
                vmin=-1.0,
                vmax=1.0,
            )

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        color_bar = fig.colorbar(im1, cax=cax, orientation="vertical")
        color_bar.set_alpha(1)
        color_bar.draw_all()
        plt.draw()

    def plot_correspondences_interactive(self):
        """
        Plots the interactive correspondence figure and updates according to user input.
        """
        img_a = load_image_to_tensor(self.cfg.image_a_path, self.cfg.display_resolution)
        image_b_path = self.cfg.image_b_path
        if image_b_path is None:
            image_b_path = self.cfg.image_a_path
        img_b = load_image_to_tensor(
            image_b_path,
            self.cfg.display_resolution,
            self.cfg.brightness_factor,
            self.cfg.contrast_factor,
            self.cfg.saturation_factor,
            self.cfg.hue_factor,
            self.cfg.gaussian_sigma,
            self.cfg.gaussian_kernel_size,
        )

        fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
        self.reset_axes(axes)
        fig.tight_layout()

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x = (event.xdata - self.cfg.display_resolution / 2) / (self.cfg.display_resolution / 2)
                y = (event.ydata - self.cfg.display_resolution / 2) / (self.cfg.display_resolution / 2)
                query_point = torch.tensor([[x, y]]).float().reshape(1, 1, 1, 2).cuda()
                self.plot_figure(img_a, img_b, query_point, axes, fig)

        fig.canvas.mpl_connect("button_press_event", onclick)
        query_point = torch.tensor([[0.0, 0.0]]).reshape(1, 1, 1, 2).cuda()
        self.plot_figure(img_a, img_b, query_point, axes, fig)
        plt.show()

    def get_net_fd(self, feats1, feats2, label1, label2, coords1, coords2):
        with torch.no_grad():
            feat_samples1 = sample(feats1, coords1)
            feat_samples2 = sample(feats2, coords2)
            label_samples1 = sample(
                F.one_hot(label1 + 1, self.n_classes + 1).to(torch.float).permute(0, 3, 1, 2),
                coords1,
            )
            label_samples2 = sample(
                F.one_hot(label2 + 1, self.n_classes + 1).to(torch.float).permute(0, 3, 1, 2),
                coords2,
            )
            fd = tensor_correlation(norm(feat_samples1), norm(feat_samples2))
            ld = tensor_correlation(label_samples1, label_samples2)
        return ld, fd, label_samples1.argmax(1), label_samples2.argmax(1)

    def prep_fd(self, fd):
        fd -= fd.min()
        fd /= fd.max()
        return fd.reshape(-1)

    def generate_pr_plot(self, preds, targets, name):
        preds = preds.cpu().reshape(-1)
        preds -= preds.min()
        preds /= preds.max()
        targets = targets.to(torch.int64).cpu().reshape(-1)
        precisions, recalls, _ = precision_recall_curve(targets, preds)
        average_precision = average_precision_score(targets, preds)
        data = {
            "precisions": precisions,
            "recalls": recalls,
            "average_precision": average_precision,
            "name": name,
        }
        with open(
            join(
                self.cfg.pr_output_data_dir,
                self.cfg.dataset_name
                + "_"
                + self.stego.full_backbone_name
                + "_"
                + str(self.cfg.pr_resolution)
                + ".pkl",
            ),
            "wb",
        ) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plt.plot(
            recalls,
            precisions,
            label="AP={}% {}".format(int(average_precision * 100), name),
        )

    def plot_pr(self):
        self.n_classes = 256  # model.n_classes
        if self.cfg.plot_stego_pr or self.cfg.plot_backbone_pr:
            val_loader_crop = "center"
            val_dataset = ContrastiveSegDataset(
                data_dir=self.cfg.data_dir,
                dataset_name=self.cfg.dataset_name,
                image_set="val",
                transform=get_transform(self.cfg.pr_resolution, False, val_loader_crop),
                target_transform=get_transform(self.cfg.pr_resolution, True, val_loader_crop),
                model_type=self.stego.backbone_name,
                resolution=self.cfg.pr_resolution,
                mask=True,
                pos_images=True,
                pos_labels=True,
            )
            print("Calculating PR curves for {} with model {}".format(self.cfg.dataset_name, self.cfg.model_path))
            lds = []
            backbone_fds = []
            stego_fds = []
            for data in tqdm(val_dataset):
                img = torch.unsqueeze(data["img"], dim=0).cuda()
                label = data["label"].cuda()
                feats, code = self.stego.forward(img)
                coord_shape = [
                    img.shape[0],
                    self.stego.cfg.feature_samples,
                    self.stego.cfg.feature_samples,
                    2,
                ]
                coords1 = torch.rand(coord_shape, device=img.device) * 2 - 1
                coords2 = torch.rand(coord_shape, device=img.device) * 2 - 1
                ld, stego_fd, _, _ = self.get_net_fd(code, code, label, label, coords1, coords2)
                ld, backbone_fd, _, _ = self.get_net_fd(feats, feats, label, label, coords1, coords2)
                lds.append(ld)
                backbone_fds.append(backbone_fd)
                stego_fds.append(stego_fd)
            ld = torch.cat(lds, dim=0)
            backbone_fd = torch.cat(backbone_fds, dim=0)
            stego_fd = torch.cat(stego_fds, dim=0)
            if self.cfg.plot_stego_pr:
                self.generate_pr_plot(self.prep_fd(stego_fd), ld, "Stego")
            if self.cfg.plot_backbone_pr:
                self.generate_pr_plot(self.prep_fd(backbone_fd), ld, self.stego.full_backbone_name)
        for filename in self.cfg.additional_pr_curves:
            with open(join(self.cfg.pr_output_data_dir, filename), "rb") as handle:
                data = pickle.load(handle)
                plt.plot(
                    data["recalls"],
                    data["precisions"],
                    label="AP={}% {}".format(int(data["average_precision"] * 100), data["name"]),
                )
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend(fontsize=12)
        plt.ylabel("Precision", fontsize=16)
        plt.xlabel("Recall", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            join(
                self.cfg.pr_output_dir,
                self.cfg.dataset_name + "_" + self.stego.full_backbone_name + ".png",
            )
        )
        plt.show()

    def plot(self):
        if self.cfg.plot_correspondences_interactive:
            self.plot_correspondences_interactive()
        if self.cfg.plot_pr:
            plt.switch_backend("agg")
            self.plot_pr()


@hydra.main(config_path="cfg", config_name="plot_config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    plotter = Plotter(cfg)
    plotter.plot()


if __name__ == "__main__":
    prep_args()
    my_app()
