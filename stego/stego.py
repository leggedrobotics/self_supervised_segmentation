import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torchvision.transforms.functional as VF
import omegaconf
import os

from stego.backbones.backbone import *
from stego.utils import *
from stego.data import *


class SegmentationHead(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Conv2d(input_dim, dim, (1, 1)))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, input_dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(input_dim, dim, (1, 1)))

    def forward(self, inputs):
        return self.linear(inputs) + self.nonlinear(inputs)


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)


class CRF():
    def __init__(self, cfg):
        self.cfg = cfg

    def dense_crf(self, image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor) -> torch.FloatTensor:
        image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
        H, W = image.shape[:2]
        image = np.ascontiguousarray(image)

        output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                    align_corners=False).squeeze()
        output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

        c = output_probs.shape[0]
        h = output_probs.shape[1]
        w = output_probs.shape[2]

        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.cfg.pos_xy_std, compat=self.cfg.pos_w)
        d.addPairwiseBilateral(sxy=self.cfg.bi_xy_std, srgb=self.cfg.bi_rgb_std, rgbim=image, compat=self.cfg.bi_w)

        Q = d.inference(self.cfg.crf_max_iter)
        Q = np.array(Q).reshape((c, h, w))
        return torch.from_numpy(Q)    



class STEGO(pl.LightningModule):

    def __init__(self, n_classes, cfg=None):
        super().__init__()
        if cfg is None:
            with open(os.path.join(os.path.dirname(__file__), "cfg/model_config.yaml"), "r") as file:
                self.cfg = omegaconf.OmegaConf.load(file)
                cfg = self.cfg
        else:
            self.cfg = cfg
        self.dim = self.cfg.dim
        self.automatic_optimization = False
        self.n_classes = n_classes
        self.backbone_name = self.cfg.backbone
        self.backbone = get_backbone(self.cfg)
        self.backbone.eval()
        self.backbone_dim = self.backbone.get_output_feat_dim()
        self.segmentation_head = SegmentationHead(self.backbone_dim, self.dim)

        self.cluster_probe = ClusterLookup(self.dim, self.n_classes + self.cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(self.dim, n_classes, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, self.cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(self.cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.crf = CRF(self.cfg)

        self.save_hyperparameters()

    def configure_optimizers(self):
        main_params = list(self.backbone.parameters()) + list(self.segmentation_head.parameters())
        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)
        return net_optim, linear_probe_optim, cluster_probe_optim


    def forward(self, img):
        backbone_feats = self.backbone(img)
        return backbone_feats, self.segmentation_head(backbone_feats)

    def get_code(self, img):
        code1 = self.forward(img)[1]
        code2 = self.forward(img.flip(dims=[3]))[1]
        code = (code1 + code2.flip(dims=[3]))/2
        return code

    def postprocess(self, code, use_crf=True):
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        cluster_probs = self.cluster_probe(code, 2, log_probs=True)
        linear_probs = torch.log_softmax(self.linear_probe(code), dim=1)
        if use_crf:
            cluster_probs = cluster_probs.cpu()
            linear_probs = linear_probs.cpu()
            cluster_crf = torch.empty(torch.Size(img.size()[:-3]+img.size()[-2:]))
            linear_crf = torch.empty(torch.Size(img.size()[:-3]+img.size()[-2:]))
            for j in range(img.shape[0]):
                single_img = img[j]
                x = self.crf.dense_crf(single_img, cluster_probs[j]).argmax(0)
                cluster_crf[j] = x
                linear_crf[j] = self.crf.dense_crf(single_img, linear_probs[j]).argmax(0)
            cluster_preds = cluster_crf.int()
            linear_preds = linear_crf.int()
        else:
            linear_preds = linear_probs.argmax(1)
            cluster_preds = cluster_probs.argmax(1)
        return cluster_preds.to(self.device), linear_preds.to(self.device)



    def training_step(self, batch, batch_idx):
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        log_args = dict(sync_dist=False, rank_zero_only=True)

        with torch.no_grad():
            img = batch["img"]
            img_pos = batch["img_pos"]
            label = batch["label"]

        feats, code = self.forward(img)
        feats_pos, code_pos = self.forward(img_pos)

        (
            pos_intra_loss, pos_intra_cd,
            pos_inter_loss, pos_inter_cd,
            neg_inter_loss, neg_inter_cd,
        ) = self.contrastive_corr_loss_fn(
            feats, feats_pos,
            code, code_pos,
        )
        neg_inter_loss = neg_inter_loss.mean()
        pos_intra_loss = pos_intra_loss.mean()
        pos_inter_loss = pos_inter_loss.mean()

        self.log('loss/pos_intra', pos_intra_loss)
        self.log('loss/pos_inter', pos_inter_loss)
        self.log('loss/neg_inter', neg_inter_loss)
        self.log('cd/pos_intra', pos_intra_cd.mean())
        self.log('cd/pos_inter', pos_inter_cd.mean())
        self.log('cd/neg_inter', neg_inter_cd.mean())

        loss = (self.cfg.pos_inter_weight * pos_inter_loss +
                    self.cfg.pos_intra_weight * pos_intra_loss +
                    self.cfg.neg_inter_weight * neg_inter_loss)

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss

        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        return loss


    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]

        with torch.no_grad():
            code = self.forward(img)[1]
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            linear_metrics = self.linear_metrics.compute()
            cluster_metrics = self.cluster_metrics.compute()

            self.log('val/linear/mIoU', linear_metrics['test/linear/mIoU'])
            self.log('val/linear/Accuracy', linear_metrics['test/linear/Accuracy'])
            self.log('val/cluster/mIoU', cluster_metrics['test/cluster/mIoU'])
            self.log('val/cluster/Accuracy', cluster_metrics['test/cluster/Accuracy'])


            return {
                'img': img[:self.cfg.val_n_imgs].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.val_n_imgs].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.val_n_imgs].detach().cpu(),
                "label": label[:self.cfg.val_n_imgs].detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            self.linear_metrics.reset()
            self.cluster_metrics.reset()