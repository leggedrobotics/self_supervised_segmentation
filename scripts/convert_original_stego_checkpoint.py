#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
############################################
# Conversion of original STEGO checkpoints
#
# This script can be used to convert the original STEGO checkpoints, trained by Hamilton et al., to checkpoints that can be used in this package.
#
# Original checkpoints can be downloaded with download_stego_models.py
#
# Before running this script, adjust paths in the config file cfg/convert_checkpoint_config.yaml
#
############################################


import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset
import torch.multiprocessing
from torch import nn
import os
import omegaconf
import copy

import stego.backbones.dino.vision_transformer as vits
from stego.utils import UnsupervisedMetrics, prep_args
from stego.modules import ClusterLookup, ContrastiveCorrelationLoss
from stego.stego import Stego


class RandomDataset(Dataset):
    def __init__(self, length: int, size: tuple):
        self.len = length
        self.data = torch.randn(length, *size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class DinoFeaturizer(nn.Module):
    """
    Class from the original STEGO package, used to load the original checkpoint.
    """

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=0.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            # state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
            # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights found at {} and loaded with msg: {}".format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
        )

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert img.shape[2] % self.patch_size == 0
            assert img.shape[3] % self.patch_size == 0

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code


class ContrastiveCRFLoss(nn.Module):
    """
    Class from the original STEGO package, used to load the original checkpoint.
    """

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert guidance.shape[0] == clusters.shape[0]
        assert guidance.shape[2:] == clusters.shape[2:]
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat(
            [
                torch.randint(0, h, size=[1, self.n_samples], device=device),
                torch.randint(0, w, size=[1, self.n_samples], device=device),
            ],
            0,
        )

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = (
            self.w1 * torch.exp(-coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta))
            + self.w2 * torch.exp(-coord_diff / (2 * self.gamma))
            - self.shift
        )

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)


class LitUnsupervisedSegmenter(pl.LightningModule):
    """
    Class from the original STEGO package, used to load the original checkpoint.
    """

    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        dim = cfg.dim
        self.net = DinoFeaturizer(dim, cfg)
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics("test/linear/", n_classes, 0, False)
        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics("final/linear/", n_classes, 0, False)
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift
        )
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False
        self.automatic_optimization = False
        self.label_cmap = None
        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]


@hydra.main(config_path="cfg", config_name="convert_checkpoint_config.yaml")
def my_app(cfg: DictConfig) -> None:
    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    with open(os.path.join(os.path.dirname(__file__), "../stego/cfg/model_config.yaml"), "r") as file:
        model_cfg = omegaconf.OmegaConf.load(file)
    model_cfg.backbone = model.cfg.arch
    model_cfg.backbone_type = model.cfg.model_type
    model_cfg.patch_size = model.cfg.dino_patch_size
    model_cfg.dim = model.cfg.dim
    model_cfg.extra_clusters = model.cfg.extra_clusters
    n_classes = model.n_classes
    stego = Stego(n_classes, model_cfg)

    with torch.no_grad():
        stego.cluster_probe = copy.deepcopy(model.cluster_probe)
        stego.linear_probe = copy.deepcopy(model.linear_probe)
        stego.segmentation_head.linear = copy.deepcopy(model.net.cluster1)
        stego.segmentation_head.nonlinear = copy.deepcopy(model.net.cluster2)

    trainer = Trainer(enable_checkpointing=False, max_steps=0)
    trainer.predict(stego, RandomDataset(1, (1, 3, 224, 224)))
    trainer.save_checkpoint(cfg.output_path)


if __name__ == "__main__":
    prep_args()
    my_app()
