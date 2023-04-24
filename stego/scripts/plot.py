import os
from os.path import join
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms as T
from mpl_toolkits.axes_grid1 import make_axes_locatable

from stego.stego.stego import *



class Plotter():
    def __init__(self, cfg):
        self.cfg = cfg
        self.stego = STEGO(cfg.n_classes).cuda()

    def reset_axes(self, axes):
        axes[0].clear()
        remove_axes(axes)
        axes[0].set_title("Image A and Query Point", fontsize=20)
        axes[1].set_title("Feature Cosine Similarity", fontsize=20)
        axes[2].set_title("Image B", fontsize=20)


    def get_heatmaps(self, img, img_pos, query_points, zero_mean=True, zero_clamp=True):
        feats1, _ = self.stego.get_feats(img.cuda())
        feats2, _ = self.stego.get_feats(img_pos.cuda())

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

        heatmap_intra = F.interpolate(
            attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
        heatmap_inter = F.interpolate(
            attn_inter, img_pos.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()

        return heatmap_intra, heatmap_inter


    def plot_figure(self, img_a, img_b, query_point, axes, fig):
        _, heatmap_correspondence = self.get_heatmaps(img_a, img_b, query_point, zero_mean=self.cfg.zero_mean, zero_clamp=self.cfg.zero_clamp)
        point = ((query_point[0, 0, 0] + 1) / 2 * self.cfg.resolution).cpu()
        self.reset_axes(axes)
        axes[0].imshow(prep_for_plot(img_a[0], rescale=False))
        axes[2].imshow(prep_for_plot(img_b[0], rescale=False))
        axes[0].scatter(point[0], point[1], color=(1, 0, 0), marker="x", s=500, linewidths=5)

        img_b_bw = prep_for_plot(img_b[0], rescale=False) * .8
        img_b_bw = np.ones_like(img_b_bw) * np.expand_dims(np.dot(np.array(img_b_bw)[..., :3], [0.2989, 0.5870, 0.1140]), -1)
        axes[1].imshow(img_b_bw)
        im1 = None
        if self.cfg.zero_clamp:
            im1 = axes[1].imshow(heatmap_correspondence[0], alpha=0.5, cmap=self.cfg.cmap, vmin=0.0, vmax=1.0)
        else:
            im1 = axes[1].imshow(heatmap_correspondence[0], alpha=0.5, cmap=self.cfg.cmap, vmin=-1.0, vmax=1.0)

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        color_bar = fig.colorbar(im1, cax=cax, orientation='vertical')
        color_bar.set_alpha(1)
        color_bar.draw_all()
        plt.show()


    def plot_correspondences_interactive(self):
        img_a = load_image_to_tensor(self.cfg.image_a_path, self.cfg.resolution)
        image_b_path = self.cfg.image_b_path
        if image_b_path is None:
            image_b_path = self.cfg.image_a_path
        img_b = load_image_to_tensor(image_b_path, self.cfg.resolution, self.cfg.brightness_factor,
                                     self.cfg.contrast_factor, self.cfg.saturation_factor,
                                     self.cfg.hue_factor, self.cfg.gaussian_sigma, self.cfg.gaussian_kernel_size)

        fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
        self.reset_axes(axes)
        fig.tight_layout()

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x = (event.xdata - self.cfg.resolution/2) / (self.cfg.resolution/2)
                y = (event.ydata - self.cfg.resolution/2) / (self.cfg.resolution/2)
                query_point = torch.tensor([[x, y]]).float().reshape(1, 1, 1, 2).cuda()
                self.plot_figure(img_a, img_b, query_point, axes, fig)

        fig.canvas.mpl_connect('button_press_event', onclick)
        query_point = torch.tensor([[0.0, 0.0]]).reshape(1, 1, 1, 2).cuda()
        self.plot_figure(img_a, img_b, query_point, axes, fig)
    
    def plot(self):
        if self.cfg.plot_correspondences_interactive:
            self.plot_correspondences_interactive()



@hydra.main(config_path="cfg", config_name="plot_config.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    plotter = Plotter(cfg)
    plotter.plot()


if __name__ == "__main__":
    prep_args()
    my_app()