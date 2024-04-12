#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import collections
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import (
    np_str_obj_array_pattern,
    default_collate_err_msg_format,
)
from torchvision import transforms as T
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from PIL import Image
import sys
import os


def load_image_to_tensor(
    img_path,
    resolution=320,
    brightness_factor=1.0,
    contrast_factor=1.0,
    saturation_factor=1.0,
    hue_factor=0.0,
    gaussian_sigma=None,
    gaussian_kernel_size=None,
):
    img = Image.open(img_path)
    transforms = []
    if brightness_factor != 1.0 or contrast_factor != 1.0 or saturation_factor != 1.0 or hue_factor != 0.0:
        transforms.append(
            T.ColorJitter(
                brightness=(brightness_factor, brightness_factor),
                contrast=(contrast_factor, contrast_factor),
                saturation=(saturation_factor, saturation_factor),
                hue=(hue_factor, hue_factor),
            )
        )
    if gaussian_sigma is not None and gaussian_kernel_size is not None:
        transforms.append(T.GaussianBlur(kernel_size=gaussian_kernel_size, sigma=gaussian_sigma))
    elif gaussian_sigma is not None and gaussian_kernel_size is not None:
        raise ValueError(
            "Both sigma and kernel size for gaussian blur augmentation need to be None or specified, but exactly one was specified."
        )
    transforms.append(get_transform(resolution, False, "center"))
    preprocess_transform = T.Compose(transforms)
    image_tensor = torch.unsqueeze(preprocess_transform(img), 0)
    return image_tensor


def get_nn_file_name(data_dir, dataset_name, model_type, image_set, resolution):
    return os.path.join(
        data_dir,
        dataset_name,
        "nns",
        "nns_{}_{}_{}.npz".format(model_type, image_set, resolution),
    )


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def get_transform(res, is_label, crop_type, is_tensor=False, do_normalize=True):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    transform = [T.Resize(res, T.InterpolationMode.NEAREST), cropper]

    if is_label:
        transform.append(ToTargetTensor())
    else:
        if not is_tensor:
            transform.append(T.ToTensor())

        if do_normalize:
            transform.append(normalize)

    return T.Compose(transform)


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode="border", align_corners=True)


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def prep_args():
    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def plot_distributions(value_lists, n_bins, names, x_name, output_file):
    plt.clf()
    plt.xlabel(x_name)
    plt.ylabel("Frequency")
    plt.title("Distribution of {}".format(x_name))
    for i, values in enumerate(value_lists):
        if len(values) == 0:
            continue
        values_np = np.array(values)
        hist, bin_edges = np.histogram(
            values_np,
            bins=np.linspace(np.min(values_np), np.max(values_np), num=n_bins + 1),
            density=True,
        )
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(x, hist, label=names[i])
    plt.legend()
    plt.savefig(output_file)


class UnsupervisedMetrics(Metric):
    def __init__(
        self,
        prefix: str,
        n_classes: int,
        extra_clusters: int,
        compute_hungarian: bool,
        dist_sync_on_step=True,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state(
            "stats",
            default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (
                (actual >= 0)
                & (actual < self.n_classes)
                & (preds >= 0)
                & (preds < self.n_classes + self.extra_clusters)
            )
            actual = actual[mask]
            preds = preds[mask]
            self.stats += (
                torch.bincount(
                    (self.n_classes + self.extra_clusters) * actual + preds,
                    minlength=self.n_classes * (self.n_classes + self.extra_clusters),
                )
                .reshape(self.n_classes, self.n_classes + self.extra_clusters)
                .t()
                .to(self.stats.device)
            )

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (
                torch.arange(self.n_classes).unsqueeze(1),
                torch.arange(self.n_classes).unsqueeze(1),
            )
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        # prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {
            self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
            self.prefix + "Accuracy": opc.item(),
        }
        return {k: 100 * v for k, v in metric_dict.items()}


class WVNMetrics(Metric):
    def __init__(
        self,
        prefix: str,
        n_clusters: int,
        dist_sync_on_step=True,
        save_plots=False,
        output_dir=None,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.prefix = prefix
        self.n_clusters = n_clusters
        self.output_dir = None
        if save_plots:
            self.output_dir = output_dir
        # TN, FP, FN, TP for the traversable class (1)
        self.add_state("stats", default=torch.zeros(4, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("feature_var", default=[], dist_reduce_fx="cat")
        self.add_state("code_var", default=[], dist_reduce_fx="cat")
        self.add_state("avg_n_clusters", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")

    def update(
        self,
        clusters: torch.Tensor,
        target: torch.Tensor,
        features: torch.Tensor,
        code: torch.Tensor,
        time: float,
    ):
        with torch.no_grad():
            actual = target.reshape(-1)
            pred_clusters = clusters.reshape(-1)
            preds = self.assign_pred_to_clusters(pred_clusters, actual)
            self.stats += torch.bincount(2 * actual + preds, minlength=4).to(self.stats.device)
            cluster_count = torch.unique(pred_clusters).size(0)
            self.avg_n_clusters.append(cluster_count)
            self.time.append(time)
            if features is not None:
                self.feature_var.extend(self.update_variance(clusters, features))
            if code is not None:
                self.code_var.extend(self.update_variance(clusters, code))

    def update_variance(self, clusters: torch.Tensor, features: torch.Tensor):
        upsampled_features = F.interpolate(features, clusters.shape[-2:], mode="bilinear", align_corners=False).permute(
            0, 2, 3, 1
        )
        mean_feature_vars = []
        for i in range(self.n_clusters):
            mask = clusters == i
            if mask.shape[0] != 1:
                mask = mask.unsqueeze(0)
            cluster_features = upsampled_features[mask].reshape(-1, upsampled_features.shape[-1])
            if cluster_features.shape[0] > 1:
                mean_feature_vars.append(torch.mean(torch.var(cluster_features, dim=0)).item())
        return mean_feature_vars

    def assign_pred_to_clusters(self, clusters: torch.Tensor, target: torch.Tensor):
        counts = torch.zeros(2, self.n_clusters, dtype=torch.int64)
        for i in range(2):
            mask = target == i
            counts[i] = torch.bincount(clusters[mask], minlength=self.n_clusters)
        cluster_pred = torch.where(counts[0] > counts[1], 0, 1)
        pred = cluster_pred[clusters.long()]
        return pred

    def compute_list_metric(self, metric_name, values, metric_dict, print_metrics=False):
        if len(values) == 0:
            return
        value = np.mean(np.array(values))
        metric_dict[self.prefix + "/" + metric_name] = value
        if print_metrics:
            print("\t{}: {}".format(metric_name, value))
        if self.output_dir is not None:
            plot_distributions(
                [values],
                100,
                [self.prefix],
                metric_name,
                os.path.join(self.output_dir, self.prefix + "_" + metric_name + ".png"),
            )

    def compute(self, print_metrics=False):
        tn = self.stats[0]
        fp = self.stats[1]
        fn = self.stats[2]
        tp = self.stats[3]

        iou = tp / (tp + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)

        metric_dict = {
            self.prefix + "/IoU": iou.item(),
            self.prefix + "/Accuracy": acc.item(),
        }

        if print_metrics:
            print(self.prefix)
            print("\tIoU: {}".format(iou.item()))
            print("\tAccuracy: {}".format(acc.item()))

        self.compute_list_metric("Avg_clusters", self.avg_n_clusters, metric_dict, print_metrics)
        self.compute_list_metric("Feature_var", self.feature_var, metric_dict, print_metrics)
        self.compute_list_metric("Code_var", self.code_var, metric_dict, print_metrics)
        self.compute_list_metric("Time", self.time, metric_dict, print_metrics)

        values_dict = {
            "Avg_clusters": self.avg_n_clusters,
            "Feature_var": self.feature_var,
            "Code_var": self.code_var,
            "Time": self.time,
        }

        return metric_dict, values_dict


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
