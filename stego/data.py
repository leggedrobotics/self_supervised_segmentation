#
# Copyright (c) Mark Hamilton. All rights reserved.
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
import os

from stego.utils import get_nn_file_name


class UnlabeledImageFolder(Dataset):
    """
    A simple Dataset class to read images from a given folder.
    """

    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = root
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.images[index])).convert("RGB")
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


def create_cityscapes_colormap():
    colors = [
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 0),
    ]
    return np.array(colors)


class DirectoryDataset(Dataset):
    """
    A Dataset class that reads images and (if available) labels from the given directory.
    The expected structure of the directory:
    data_dir
    |-- dataset_name
        |-- imgs
            |-- image_set
        |-- labels
            |-- image_set

    If available, file names in labels/image_set should be the same as file names in imgs/image_set (excluding extensions).
    If labels are not available (there is no labels folder) this class returns zero arrays of shape corresponding to the image shape.
    """

    def __init__(self, data_dir, dataset_name, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        self.dataset_name = dataset_name
        self.dir = os.path.join(data_dir, dataset_name)
        self.img_dir = os.path.join(self.dir, "imgs", self.split)
        self.label_dir = os.path.join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert len(self.img_files) > 0, "Could not find any images in dataset directory {}".format(self.img_dir)
        if os.path.exists(os.path.join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(
                self.label_files
            ), "The {} dataset contains a different number of images and labels: {} images and {} labels".format(
                self.dataset_name, len(self.img_files), len(self.label_files)
            )
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_name = self.img_files[index]
        img = Image.open(os.path.join(self.img_dir, image_name))
        if self.label_files is not None:
            label_name = self.label_files[index]
            label = Image.open(os.path.join(self.label_dir, label_name))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)
        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64)

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class ContrastiveSegDataset(Dataset):
    """
    The main Dataset class used by STEGO.
    Internally uses the DirectoryDataset class to load images.
    Additionally, this class uses the precomputed Nearest Neighbor files to extract the knn corresponding image for STEGO training.
    It returns a dictionary containing an image and its positive pair (one of the nearest neighbor images).
    """

    def __init__(
        self,
        data_dir,
        dataset_name,
        image_set,
        transform,
        target_transform,
        model_type,
        resolution,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        num_neighbors=5,
        mask=False,
        pos_labels=False,
        pos_images=False,
        extra_transform=None,
    ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform
        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = DirectoryDataset(data_dir, dataset_name, image_set, transform, target_transform)

        feature_cache_file = get_nn_file_name(data_dir, dataset_name, model_type, image_set, resolution)
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file):
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert (
                len(self.dataset) == self.nns.shape[0]
            ), "Found different numbers of images in dataset {} and nn file {}".format(dataset_name, feature_cache_file)

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        coord_entries = torch.meshgrid(
            [
                torch.linspace(-1, 1, pack[0].shape[1]),
                torch.linspace(-1, 1, pack[0].shape[2]),
            ]
        )
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
        }

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        if self.aug_photometric_transform is not None:
            img_aug = self.aug_photometric_transform(self.aug_geometric_transform(pack[0]))

            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)

            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret
