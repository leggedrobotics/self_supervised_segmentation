import torch

import torch.nn.functional as F
from .dino import vision_transformer as vits
from torch import nn
import numpy as np


def get_backbone(cfg):
    if cfg.backbone == "dino":
        return DinoViT(cfg)
    else:
        raise ValueError("Backbone {} unavailable".format(cfg.backbone))


class DinoViT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = self.cfg.patch_size
        self.model_type = self.cfg.backbone_type
        self.model = vits.__dict__[self.model_type](
            patch_size=self.patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=np.clip(self.cfg.dropout_p, 0.0, 1.0))

        if self.model_type == "vit_small" and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif self.model_type == "vit_small" and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif self.model_type == "vit_base" and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif self.model_type == "vit_base" and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Model type {} unavailable with patch size {}".format(self.model_type, self.patch_size))

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if self.model_type == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

    def get_output_feat_dim(self):
        return self.n_feats

    def forward(self, img):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        if self.cfg.dropout_p > 0:
            return self.dropout(image_feat)
        else:
            return image_feat