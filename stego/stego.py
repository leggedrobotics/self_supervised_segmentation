#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import omegaconf
import os
import wandb
import matplotlib as plt
import io


from stego.backbones.backbone import get_backbone
from stego.utils import UnsupervisedMetrics
from stego.data import Image
from stego.modules import SegmentationHead, ClusterLookup, ContrastiveCorrelationLoss, CRF, KMeans


class Stego(pl.LightningModule):
    """
    The main STEGO class.
    """

    def __init__(self, n_classes, n_image_clusters=0, cfg=None):
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
        self.backbone = get_backbone(self.cfg)
        self.full_backbone_name = self.backbone.get_backbone_name()
        self.backbone.eval()
        self.backbone_dim = self.backbone.get_output_feat_dim()
        self.segmentation_head = SegmentationHead(self.backbone_dim, self.dim)

        self.cluster_probe = ClusterLookup(self.dim, self.n_classes + self.cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(self.dim, n_classes, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, self.cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics("test/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(self.cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.crf = CRF(self.cfg)

        self.n_image_clusters = n_image_clusters
        if n_image_clusters == 0:
            self.n_image_clusters = n_classes
        self.kmeans = KMeans(
            num_clusters=self.n_image_clusters,
            cluster_centers=None,
            distance_metric="cosine",
            max_iterations=100,
        )

        self.cd_hist = torch.zeros(40)

        self.save_hyperparameters()

    def reset_clusters(self, n_classes, extra_clusters):
        """
        Resets STEGO's cluster and linear probes, possibly with a different number of classes and extra clusters for the cluster probe.
        """
        self.cluster_probe = ClusterLookup(self.dim, n_classes + extra_clusters)
        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, extra_clusters, True)
        self.linear_probe = nn.Conv2d(self.dim, n_classes, (1, 1))
        self.linear_metrics = UnsupervisedMetrics("test/linear/", n_classes, 0, False)
        self.n_classes = n_classes

    def configure_optimizers(self):
        main_params = list(self.backbone.parameters()) + list(self.segmentation_head.parameters())
        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=self.cfg.linear_lr)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=self.cfg.cluster_lr)
        return net_optim, linear_probe_optim, cluster_probe_optim

    def forward(self, img):
        backbone_feats = self.backbone(img)
        return backbone_feats, self.segmentation_head(backbone_feats)

    def get_code(self, img):
        """
        Returns segmentation features for a given image.
        Returned features are an average of two passes through STEGO, with the input image and its horizontal flip.
        """
        # code1 = self.forward(img)[1]
        # code2 = self.forward(img.flip(dims=[3]))[1]
        # code = (code1 + code2.flip(dims=[3])) / 2
        # return code
        code = self.forward(img)[1]
        return code

    def postprocess_crf(self, img, probs):
        """
        Performs the CRF postprocessing step on the given image and a set of predicted class probabilities.
        The class probabilities are interpolated to fit the image size inside the dense_crf function.
        """
        pred = torch.empty(torch.Size(img.size()[:-3] + img.size()[-2:]))
        for j in range(img.shape[0]):
            single_img = img[j]
            x = self.crf.dense_crf(single_img, probs[j]).argmax(0)
            pred[j] = x
        return pred.int()

    def postprocess_cluster(self, code, img, use_crf=True, image_clustering=False):
        """
        Cluster probe postprocessing of STEGO.
        For the given features, the cluster probe is run, followed by CRF (if enabled).
        If enabled, performs the K-means clustering only of the segmentation features in the given batch.

        Arguments:
        - code - STEGO's segmentation features.
        - img - input image.
        - use_crf - enables CRF on the image and class probabilities from the cluster probe.
        - image_clustering - enables per-image clustering. If True, STEGO's cluster probe is ignored and K-means is run on the given segmentation features to produce the cluster probabilities,
        """
        orig_code = code.permute(0, 2, 3, 1)
        code = F.interpolate(code, img.shape[-2:], mode="bilinear", align_corners=False)
        if image_clustering:
            self.kmeans.fit(orig_code.reshape((-1, orig_code.shape[-1])))

            normed_centers = F.normalize(self.kmeans.final_cluster_centers, dim=1)
            normed_code = F.normalize(code, dim=1).permute(0, 2, 3, 1)
            inner_products = torch.einsum("bhwc,nc->bnhw", normed_code, normed_centers)
            cluster_probs = nn.functional.softmax(inner_products * 2, dim=1)
        else:
            cluster_probs = self.cluster_probe(code, 2, log_probs=True)
        if use_crf:
            cluster_preds = self.postprocess_crf(img, cluster_probs)
        else:
            cluster_preds = cluster_probs.argmax(dim=1)
        return cluster_preds

    def postprocess_linear(self, code, img, use_crf=True):
        """
        Linear probe postprocessing of STEGO.
        For the given features, the linear probe is run, followed by CRF (if enabled).

        Arguments:
        - code - STEGO's segmentation features.
        - img - input image.
        - use_crf - enables CRF on the image and class probabilities from the linear probe.
        """
        code = F.interpolate(code, img.shape[-2:], mode="bilinear", align_corners=False)
        linear_probs = torch.log_softmax(self.linear_probe(code), dim=1)
        if use_crf:
            linear_preds = self.postprocess_crf(img, linear_probs)
        else:
            linear_preds = linear_probs.argmax(1)
        return linear_preds

    def postprocess(
        self,
        code,
        img,
        use_crf_cluster=True,
        use_crf_linear=True,
        image_clustering=False,
    ):
        """
        Complete postprocessing of STEGO.
        For the given features, both the cluster and linear probes are run, followed by CRF (if enabled).
        If enabled, performs the K-means clustering only of the segmentation features in the given batch.

        Arguments:
        - code - STEGO's segmentation features.
        - img - input image.
        - use_crf_cluster - enables CRF on the image and class probabilities from the cluster probe.
        - use_crf_linear - enables CRF on the image and class probabilities from the linear probe.
        - image_clustering - enables per-image clustering. If True, STEGO's cluster probe is ignored and K-means is run on the given segmentation features to produce the cluster probabilities,
        """
        cluster_preds = self.postprocess_cluster(code, img, use_crf_cluster, image_clustering)
        linear_preds = self.postprocess_linear(code, img, use_crf_linear)
        return cluster_preds, linear_preds

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
            pos_intra_loss,
            pos_intra_cd,
            pos_inter_loss,
            pos_inter_cd,
            neg_inter_loss,
            neg_inter_cd,
        ) = self.contrastive_corr_loss_fn(
            feats,
            feats_pos,
            code,
            code_pos,
        )
        neg_inter_loss = neg_inter_loss.mean()
        pos_intra_loss = pos_intra_loss.mean()
        pos_inter_loss = pos_inter_loss.mean()

        self.cd_hist = torch.add(self.cd_hist, torch.histc(pos_intra_cd.cpu(), bins=40, min=-1, max=1))
        self.cd_hist = torch.add(self.cd_hist, torch.histc(pos_inter_cd.cpu(), bins=40, min=-1, max=1))
        self.cd_hist = torch.add(self.cd_hist, torch.histc(neg_inter_cd.cpu(), bins=40, min=-1, max=1))

        self.log("loss/pos_intra", pos_intra_loss)
        self.log("loss/pos_inter", pos_inter_loss)
        self.log("loss/neg_inter", neg_inter_loss)
        self.log("cd/pos_intra", pos_intra_cd.mean())
        self.log("cd/pos_inter", pos_inter_cd.mean())
        self.log("cd/neg_inter", neg_inter_cd.mean())

        loss = (
            self.cfg.pos_inter_weight * pos_inter_loss
            + self.cfg.pos_intra_weight * pos_intra_loss
            + self.cfg.neg_inter_weight * neg_inter_loss
        )

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode="bilinear", align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log("loss/linear", linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss

        self.log("loss/cluster", cluster_loss, **log_args)
        self.log("loss/total", loss, **log_args)

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
            code = F.interpolate(code, label.shape[-2:], mode="bilinear", align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            linear_metrics = self.linear_metrics.compute()
            cluster_metrics = self.cluster_metrics.compute()

            self.log("val/linear/mIoU", linear_metrics["test/linear/mIoU"])
            self.log("val/linear/Accuracy", linear_metrics["test/linear/Accuracy"])
            self.log("val/cluster/mIoU", cluster_metrics["test/cluster/mIoU"])
            self.log("val/cluster/Accuracy", cluster_metrics["test/cluster/Accuracy"])

            return {
                "img": img[: self.cfg.val_n_imgs].detach().cpu(),
                "linear_preds": linear_preds[: self.cfg.val_n_imgs].detach().cpu(),
                "cluster_preds": cluster_preds[: self.cfg.val_n_imgs].detach().cpu(),
                "label": label[: self.cfg.val_n_imgs].detach().cpu(),
            }

    def on_validation_epoch_end(self, outputs) -> None:
        super().on_validation_epoch_end(outputs)
        with torch.no_grad():
            self.linear_metrics.reset()
            self.cluster_metrics.reset()

        for i in range(self.cfg.val_n_imgs):
            img = outputs[0]["img"][i].cpu().numpy().transpose((1, 2, 0))
            label = torch.squeeze(outputs[0]["label"][i]).cpu().numpy()
            cluster = torch.squeeze(outputs[0]["cluster_preds"][i]).cpu().numpy()
            linear = torch.squeeze(outputs[0]["linear_preds"][i]).cpu().numpy()
            vis = wandb.Image(
                img,
                masks={
                    "label": {"mask_data": label},
                    "cluster": {"mask_data": cluster},
                    "linear": {"mask_data": linear},
                },
                caption="Image" + str(i),
            )
            self.logger.experiment.log({"Image" + str(i): vis})

        self.cd_hist = self.cd_hist / torch.sum(self.cd_hist)
        x = [-1 + i * (2 / 40) + 1 / 40 for i in range(40)]
        plt.figure()
        ax = plt.axes()
        ax.plot(x, self.cd_hist)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 0.4])

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        hist_img = Image.open(img_buf)
        hist_vis = wandb.Image(hist_img, caption="Learned Feature Similarity Distribution")
        self.logger.experiment.log({"Histogram": hist_vis})
        img_buf.close()
        self.cd_hist = torch.zeros(40)
