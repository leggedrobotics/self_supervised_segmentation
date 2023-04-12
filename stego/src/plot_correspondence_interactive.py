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

from stego.src.utils import get_transform, load_model, prep_for_plot, remove_axes, prep_args
from stego.src.modules import FeaturePyramidNet, DinoFeaturizer, sample
from stego.src.plot_dino_correspondence import get_heatmaps, plot_heatmap


def load_images_to_tensor(cfg):
    image_a = Image.open(cfg.image_a_path)
    image_b = image_a
    if cfg.image_b_path is not None:
        image_b = Image.open(cfg.image_b_path)
    preprocess_a_transform = get_transform(cfg.resolution, False, "center")
    preprocess_b_transform = preprocess_a_transform
    if cfg.use_augmentations:
        color_transform = T.ColorJitter(brightness=(cfg.brightness_factor, cfg.brightness_factor),
                                        contrast=(cfg.contrast_factor, cfg.contrast_factor),
                                        saturation=(cfg.saturation_factor, cfg.saturation_factor),
                                        hue=(cfg.hue_factor, cfg.hue_factor))
        if cfg.use_gaussian_blur:
            gaussian_transform = T.GaussianBlur(kernel_size=cfg.gaussian_kernel_size, sigma=cfg.gaussian_sigma)
            preprocess_b_transform = T.Compose([color_transform, gaussian_transform, preprocess_a_transform])
        else:
            preprocess_b_transform = T.Compose([color_transform, preprocess_a_transform])
    image_a_tensor = torch.unsqueeze(preprocess_a_transform(image_a), 0)
    image_b_tensor = torch.unsqueeze(preprocess_b_transform(image_b), 0)
    return image_a_tensor, image_b_tensor

def save_figure(cfg):
    img_a_name = os.path.splitext(os.path.basename(cfg.image_a_path))[0]
    img_b_name = img_a_name
    if cfg.image_b_path is not None:
        img_b_name = os.path.splitext(os.path.basename(cfg.image_b_path))[0]
    arch_name = cfg.arch
    if arch_name == "dino":
        arch_name = cfg.arch+"_"+cfg.model_type
    fig_name = os.path.join(cfg.output_dir, "corr_"+img_a_name+"_"+img_b_name+"_"+arch_name+".png")
    plt.savefig(fig_name)

def reset_axes(axes):
    axes[0].clear()
    remove_axes(axes)
    axes[0].set_title("Image A and Query Point", fontsize=20)
    axes[1].set_title("Feature Cosine Similarity", fontsize=20)
    axes[2].set_title("Image B", fontsize=20)
    

def plot_figure(net, img_a, img_b, query_point, axes, cfg, fig):
    _, heatmap_correspondence = get_heatmaps(net, img_a, img_b, query_point, zero_mean=cfg.zero_mean, zero_clamp=cfg.zero_clamp)
    point = ((query_point[0, 0, 0] + 1) / 2 * cfg.resolution).cpu()
    reset_axes(axes)
    axes[0].imshow(prep_for_plot(img_a[0], rescale=False))
    axes[2].imshow(prep_for_plot(img_b[0], rescale=False))
    axes[0].scatter(point[0], point[1], color=(1, 0, 0), marker="x", s=500, linewidths=5)

    img_b_bw = prep_for_plot(img_b[0], rescale=False) * .8
    img_b_bw = np.ones_like(img_b_bw) * np.expand_dims(np.dot(np.array(img_b_bw)[..., :3], [0.2989, 0.5870, 0.1140]), -1)
    axes[1].imshow(img_b_bw)
    im1 = None
    if cfg.zero_clamp:
        im1 = axes[1].imshow(heatmap_correspondence[0], alpha=0.5, cmap=cfg.cmap, vmin=0.0, vmax=1.0)
    else:
        im1 = axes[1].imshow(heatmap_correspondence[0], alpha=0.5, cmap=cfg.cmap, vmin=-1.0, vmax=1.0)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    color_bar = fig.colorbar(im1, cax=cax, orientation='vertical')
    color_bar.set_alpha(1)
    color_bar.draw_all()
    plt.show()


@hydra.main(config_path="configs", config_name="plot_interactive_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    img_a, img_b = load_images_to_tensor(cfg)

    if cfg.arch == "dino":
        net = DinoFeaturizer(cfg.dim, cfg)
    else:
        raise ValueError("Unknown arch {}".format(cfg.arch))
    net = net.cuda()

    # plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
    reset_axes(axes)
    fig.tight_layout()

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x = (event.xdata - cfg.resolution/2) / (cfg.resolution/2)
            y = (event.ydata - cfg.resolution/2) / (cfg.resolution/2)
            query_point = torch.tensor([[x, y]]).float().reshape(1, 1, 1, 2).cuda()
            plot_figure(net, img_a, img_b, query_point, axes, cfg, fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    query_point = torch.tensor([[0.0, 0.0]]).reshape(1, 1, 1, 2).cuda()
    plot_figure(net, img_a, img_b, query_point, axes, cfg, fig)


if __name__ == "__main__":
    prep_args()
    my_app()