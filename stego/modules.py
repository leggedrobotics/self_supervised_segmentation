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
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torchvision.transforms.functional as VF
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SHAPE

from stego.utils import unnorm, sample, super_perm, norm, tensor_correlation


class SegmentationHead(nn.Module):
    """
    STEGO's segmentation head module.
    """

    def __init__(self, input_dim, dim):
        super().__init__()
        self.linear = torch.nn.Sequential(torch.nn.Conv2d(input_dim, dim, (1, 1)))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, input_dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(input_dim, dim, (1, 1)),
        )

    def forward(self, inputs):
        return self.linear(inputs) + self.nonlinear(inputs)


class ClusterLookup(nn.Module):
    """
    STEGO's clustering module.
    Performs cosine distance K-means on the given features.
    """

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
            cluster_probs = (
                F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0])
                .permute(0, 3, 1, 2)
                .to(torch.float32)
            )
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class ContrastiveCorrelationLoss(nn.Module):
    """
    STEGO's correlation loss.
    """

    def __init__(
        self,
        cfg,
    ):
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

        if self.cfg.stabilize:
            loss = -cd.clamp(min_val, 0.8) * (fd - shift)
        else:
            loss = -cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(
        self,
        orig_feats: torch.Tensor,
        orig_feats_pos: torch.Tensor,
        orig_code: torch.Tensor,
        orig_code_pos: torch.Tensor,
    ):
        coord_shape = [
            orig_feats.shape[0],
            self.cfg.feature_samples,
            self.cfg.feature_samples,
            2,
        ]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (
            pos_intra_loss.mean(),
            pos_intra_cd,
            pos_inter_loss.mean(),
            pos_inter_cd,
            neg_inter_loss,
            neg_inter_cd,
        )


class CRF:
    """
    Class encapsulating STEGO's CRF postprocessing step.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def dense_crf(self, image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor) -> torch.FloatTensor:
        image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
        H, W = image.shape[:2]
        image = np.ascontiguousarray(image)

        output_logits = F.interpolate(
            output_logits.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

        c = output_probs.shape[0]
        h = output_probs.shape[1]
        w = output_probs.shape[2]

        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.cfg.pos_xy_std, compat=self.cfg.pos_w)
        d.addPairwiseBilateral(
            sxy=self.cfg.bi_xy_std,
            srgb=self.cfg.bi_rgb_std,
            rgbim=image,
            compat=self.cfg.bi_w,
        )

        Q = d.inference(self.cfg.crf_max_iter)
        Q = np.array(Q).reshape((c, h, w))
        return torch.from_numpy(Q)


class KMeans:
    """Implements the kmeans clustering algorithm in PyTorch.
    The code of this class was based on: https://github.com/kornia/kornia/pull/2304

    Args:
        num_clusters: number of clusters the data has to be assigned to
        cluster_centers: tensor of starting cluster centres can be passed instead of num_clusters
        tolerance: float value. the algorithm terminates if the shift in centers is less than tolerance
        max_iterations: number of iterations to run the algorithm for
        distance_metric: {"euclidean", "cosine"}, type of the distance metric to use
        seed: number to set torch manual seed for reproducibility
    """

    def __init__(
        self,
        num_clusters: int,
        cluster_centers: Tensor,
        tolerance: float = 10e-4,
        max_iterations: int = 0,
        distance_metric="euclidean",
        seed=None,
    ) -> None:
        KORNIA_CHECK(num_clusters != 0, "num_clusters can't be 0")

        # cluster_centers should have only 2 dimensions
        if cluster_centers is not None:
            KORNIA_CHECK_SHAPE(cluster_centers, ["C", "D"])

        self.num_clusters = num_clusters
        self.cluster_centers = cluster_centers
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if distance_metric == "euclidean":
            self._pairwise_distance = self._pairwise_euclidean_distance
        elif distance_metric == "cosine":
            self._pairwise_distance = self._pairwise_cosine_distance
        else:
            raise ValueError("Unknown distance metric")

        self.final_cluster_assignments = None
        self.final_cluster_centers = None

        if seed is not None:
            torch.manual_seed(seed)

    def get_cluster_centers(self) -> Tensor:
        KORNIA_CHECK(
            self.final_cluster_centers is not None,
            "Model has not been fit to a dataset",
        )
        return self.final_cluster_centers

    def get_cluster_assignments(self) -> Tensor:
        KORNIA_CHECK(
            self.final_cluster_assignments is not None,
            "Model has not been fit to a dataset",
        )
        return self.final_cluster_assignments

    def _initialise_cluster_centers(self, X: Tensor, num_clusters: int) -> Tensor:
        """Chooses num_cluster points from X as the initial cluster centers.

        Args:
            X: 2D input tensor to be clustered
            num_clusters: number of desired cluster centers

        Returns:
            2D Tensor with num_cluster rows
        """
        num_samples = X.shape[0]
        perm = torch.randperm(num_samples, device=X.device)
        idx = perm[:num_clusters]
        initial_state = X[idx]
        return initial_state

    def _pairwise_euclidean_distance(self, data1: Tensor, data2: Tensor) -> Tensor:
        """Computes pairwise distance between 2 sets of vectors.

        Args:
            data1: 2D tensor of shape N, D
            data2: 2D tensor of shape C, D

        Returns:
            2D tensor of shape N, C
        """
        # N*1*D
        A = data1[:, None, ...]
        # 1*C*D
        B = data2[None, ...]
        distance = (A - B) ** 2.0
        # return N*C matrix for pairwise distance
        distance = distance.sum(dim=-1)
        return distance

    def _pairwise_cosine_distance(self, data1: Tensor, data2: Tensor) -> Tensor:
        """Computes pairwise distance between 2 sets of vectors.

        Args:
            data1: 2D tensor of shape N, D
            data2: 2D tensor of shape C, D

        Returns:
            2D tensor of shape N, C
        """
        normed_A = F.normalize(data1, dim=1)
        normed_B = F.normalize(data2, dim=1)
        distance = 1.0 - torch.einsum("nd,cd->nc", normed_A, normed_B)
        return distance

    def fit(self, X: Tensor) -> None:
        """Iterative KMeans clustering till a threshold for shift in cluster centers or a maximum no of iterations
        have reached.

        Args:
            X: 2D input tensor to be clustered
        """
        KORNIA_CHECK_SHAPE(X, ["N", "D"])

        if self.cluster_centers is None:
            self.cluster_centers = self._initialise_cluster_centers(X, self.num_clusters)
        else:
            # X and cluster_centers should have same number of columns
            KORNIA_CHECK(
                X.shape[1] == self.cluster_centers.shape[1],
                f"Dimensions at position 1 of X and cluster_centers do not match. \
                {X.shape[1]} != {self.cluster_centers.shape[1]}",
            )

        current_centers = self.cluster_centers

        previous_centers = None
        iteration: int = 0

        while True:
            # find distance between X and current_centers
            distance: Tensor = self._pairwise_distance(X, current_centers)

            cluster_assignment = torch.argmin(distance, dim=1)

            previous_centers = current_centers.clone()

            one_hot_assignments = torch.nn.functional.one_hot(cluster_assignment, self.num_clusters).float()
            sum_points = torch.mm(one_hot_assignments.T, X)
            num_points = one_hot_assignments.sum(0).unsqueeze(1)

            # Handle empty clusters by replacing them with a random point
            empty_clusters = num_points.squeeze() == 0
            random_points = X[torch.randint(len(X), (torch.sum(empty_clusters),))]
            sum_points[empty_clusters, :] = random_points
            num_points[empty_clusters] = 1

            current_centers = sum_points / num_points

            # sum of distance of how much the newly computed clusters have moved from their previous positions
            center_shift = torch.sum(torch.sqrt(torch.sum((current_centers - previous_centers) ** 2, dim=1)))

            iteration = iteration + 1

            if self.tolerance is not None and center_shift**2 < self.tolerance:
                break

            if self.max_iterations != 0 and iteration >= self.max_iterations:
                break

        self.final_cluster_assignments = cluster_assignment
        self.final_cluster_centers = current_centers

    def predict(self, x: Tensor) -> Tensor:
        """Find the cluster center closest to each point in x.

        Args:
            x: 2D tensor

        Returns:
            1D tensor containing cluster id assigned to each data point in x
        """

        # x and cluster_centers should have same number of columns
        KORNIA_CHECK(
            x.shape[1] == self.final_cluster_centers.shape[1],
            f"Dimensions at position 1 of x and cluster_centers do not match. \
                {x.shape[1]} != {self.final_cluster_centers.shape[1]}",
        )

        distance = self._pairwise_distance(x, self.final_cluster_centers)
        cluster_assignment = torch.argmin(distance, axis=1)
        return cluster_assignment, distance
