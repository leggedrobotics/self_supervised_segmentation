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
from torchvision import transforms

from stego.src.utils import get_transform, load_model, prep_for_plot, remove_axes, prep_args
from stego.src.modules import FeaturePyramidNet, DinoFeaturizer, sample
from stego.src.data import ContrastiveSegDataset
from stego.src.plot_dino_correspondence import get_heatmaps, plot_heatmap


def load_images_to_tensor(cfg):
    image_a = Image.open(cfg.image_a_path)
    image_b = image_a
    if cfg.image_b_path is not None:
        image_b = Image.open(cfg.image_b_path)
    preprocess_transform = get_transform(cfg.resolution, False, "center")
    image_a_tensor = torch.unsqueeze(preprocess_transform(image_a), 0)
    image_b_tensor = torch.unsqueeze(preprocess_transform(image_b), 0)
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


@hydra.main(config_path="configs", config_name="plot_interactive_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    img_a, img_b = load_images_to_tensor(cfg)

    if cfg.arch == "dino":
        net = DinoFeaturizer(cfg.dim, cfg)
    else:
        raise ValueError("Unknown arch {}".format(cfg.arch))
    net = net.cuda()

    query_point = torch.tensor([[-.1, 0.0]]).reshape(1, 1, 1, 2).cuda()

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
    remove_axes(axes)
    axes[0].set_title("Image A and Query Point", fontsize=20)
    axes[1].set_title("Feature Cosine Similarity", fontsize=20)
    axes[2].set_title("Image B", fontsize=20)
    fig.tight_layout()

    _, heatmap_correspondence = get_heatmaps(net, img_a, img_b, query_point)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    cmaps = [
        ListedColormap([(1, 0, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 1, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 0, 1, i / 255) for i in range(255)]),
        ListedColormap([(1, 1, 0, i / 255) for i in range(255)])
    ]

    for point_num in range(query_point.shape[1]):
        point = ((query_point[0, point_num, 0] + 1) / 2 * cfg.resolution).cpu()
        img_point_h = point[0]
        img_point_w = point[1]

        plot_img = point_num == 0
        if plot_img:
            axes[0].imshow(prep_for_plot(img_a[0]))
            axes[2].imshow(prep_for_plot(img_b[0]))
        axes[0].scatter(img_point_h, img_point_w,
                        c=colors[point_num], marker="x", s=500, linewidths=5)

        plot_heatmap(axes[1], prep_for_plot(img_b[0]) * .8, heatmap_correspondence[point_num],
                        plot_img=plot_img, cmap=cmaps[point_num], symmetric=False)
    save_figure(cfg)


if __name__ == "__main__":
    prep_args()
    my_app()