import os
import random
from os.path import join

import numpy as np
import torch.multiprocessing
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
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
              (0, 0, 0)]
    return np.array(colors)

def create_freiburg_forest_colormap():
    colors = [(0, 0, 0),       # Object
              (170, 170, 170), # Trail
              (0, 255, 0),     # Grass
              (0, 120, 255),   # Sky
              (102, 102, 51)]  # Vegetation
    return np.array(colors)

def create_RUGD_colormap():
    colors = [(0, 0, 0),        # void
              (108, 64, 20),    # dirt
              (255, 229, 204),  # sand
              (0, 102, 0),      # grass
              (0, 255, 0),      # tree
              (0, 153, 153),    # pole
              (0, 128, 255),    # water
              (0, 0, 255),      # sky
              (255, 255, 0),    # vehicle
              (255, 0, 127),    # container/generic-object
              (64, 64, 64),     # asphalt
              (255, 128, 0),    # gravel
              (255, 0, 0),      # building
              (153, 76, 0),     # mulch
              (102, 102, 0),    # rock-bed 
              (102, 0, 0),      # log
              (0, 255, 128),    # bicycle
              (204, 153, 255),  # person
              (102, 0, 204),    # fence
              (255, 153, 204),  # bush
              (0, 102, 102),    # sign
              (153, 204, 255),  # rock
              (102, 255, 255),  # bridge
              (101, 101, 11),   # concrete
              (114, 85, 47)]    # picnic-table
    return np.array(colors)


def get_colormap(cfg):
    if cfg.dataset_name == "directory":
        if cfg.dir_dataset_name == "freiburg_forest":
            return create_freiburg_forest_colormap()
        elif cfg.dir_dataset_name == "RUGD":
            return create_RUGD_colormap()
        elif cfg.dir_dataset_name == "cocostuff":
            return np.array([(i, i, i) for i in range(cfg.dir_dataset_n_classes)])
    return None

def color_mask_to_class_ids(mask, cfg):
    if mask.shape[1] == 1:
        return mask.squeeze()
    cmap = get_colormap(cfg)
    if cmap is None:
        return mask
    for i in range(cmap.shape[0]):
        color = torch.from_numpy(cmap[i]).cuda()
        a = mask == color
        indices = torch.all(a, axis=3)
        mask[indices] = i
    mask = torch.unique(mask, dim=-1).squeeze()
    return mask

def cocostuff_to_27_classes(mask):
    classes27 = [
        [0],                                                    # person
        [1, 2, 3, 4, 5, 6, 7, 8],                               # vehicle
        [9, 10, 11, 12, 13, 14],                                # outdoor
        [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],               # animal
        [25, 26, 27, 28, 29, 30, 31, 32],                       # accessory
        [33, 34, 35, 36, 37, 38, 39, 40, 41, 42],               # sports
        [43, 44, 45, 46, 47, 48, 49, 50],                       # kitchen
        [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],               # food
        [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],               # furniture
        [71, 72, 73, 74, 75, 76],                               # electronic
        [77, 78, 79, 80, 81, 82],                               # appliance
        [83, 84, 85, 86, 87, 88, 89, 90],                       # indoor
        [119, 147, 154, 177, 178],                              # water
        [110, 124, 125, 135, 139, 143, 144, 146, 148, 153, 158],# ground
        [126, 134, 149, 159, 161, 181],                         # solid
        [105, 156],                                             # sky
        [93, 96, 118, 123, 128, 133, 141, 162, 168],            # plant
        [98, 112, 137, 145, 163],                               # structural
        [94, 95, 127, 150, 157, 165],                           # building
        [120, 121, 152, 169],                                   # food
        [91, 92, 103, 104, 108, 130, 136, 140, 151, 166, 167],  # textile
        [97, 106, 107, 109, 111, 122, 129, 132, 155, 160, 164], # furniture
        [179, 180],                                             # window
        [100, 113, 114, 115, 116, 117],                         # floor
        [101, 102],                                             # ceiling
        [170, 171, 172, 173, 174, 175, 176],                    # wall
        [99, 131, 138, 142],                                    # rawmaterial
    ]
    for i, class_ids in enumerate(classes27):
        for class_id in class_ids:
            mask = torch.where(mask==class_id, i, mask)
    return mask

class DirectoryDataset(Dataset):
    def __init__(self, root, path, image_set, transform, target_transform):
        super(DirectoryDataset, self).__init__()
        self.split = image_set
        self.dir = join(root, path)
        self.img_dir = join(self.dir, "imgs", self.split)
        self.label_dir = join(self.dir, "labels", self.split)

        self.transform = transform
        self.target_transform = target_transform

        self.img_files = np.array(sorted(os.listdir(self.img_dir)))
        assert len(self.img_files) > 0
        if os.path.exists(join(self.dir, "labels")):
            self.label_files = np.array(sorted(os.listdir(self.label_dir)))
            assert len(self.img_files) == len(self.label_files)
        else:
            self.label_files = None

    def __getitem__(self, index):
        image_fn = self.img_files[index]
        img = Image.open(join(self.img_dir, image_fn))
        if len(img.getbands()) == 1:
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg

        if self.label_files is not None:
            label_fn = self.label_files[index]
            label = Image.open(join(self.label_dir, label_fn))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        if self.label_files is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transform(label)
        else:
            label = torch.zeros(img.shape[1], img.shape[2], dtype=torch.int64) - 1

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.img_files)


class Potsdam(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(Potsdam, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdam")
        self.transform = transform
        self.target_transform = target_transform
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }
        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class PotsdamRaw(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(PotsdamRaw, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdamraw", "processed")
        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for im_num in range(38):
            for i_h in range(15):
                for i_w in range(15):
                    self.files.append("{}_{}_{}.mat".format(im_num, i_h, i_w))

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None):
        super(Coco, self).__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return image, target.squeeze(0), mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform):
        super(CroppedDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = image_set
        self.root = join(root, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = join(self.root, "img", self.split)
        self.label_dir = join(self.root, "label", self.split)
        self.num_images = len(os.listdir(self.img_dir))
        assert self.num_images == len(os.listdir(self.label_dir))

    def __getitem__(self, index):
        image = Image.open(join(self.img_dir, "{}.jpg".format(index))).convert('RGB')
        target = Image.open(join(self.label_dir, "{}.png".format(index)))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.target_transform(target)

        target = target - 1
        mask = target == -1
        return image, target.squeeze(0), mask

    def __len__(self):
        return self.num_images


class MaterializedDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=12, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform

        if dataset_name == "potsdam":
            self.n_classes = 3
            dataset_class = Potsdam
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "potsdamraw":
            self.n_classes = 3
            dataset_class = PotsdamRaw
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "directory":
            self.n_classes = cfg.dir_dataset_n_classes
            dataset_class = DirectoryDataset
            extra_args = dict(path=cfg.dir_dataset_name)
        elif dataset_name == "cityscapes" and crop_type is None:
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict()
        elif dataset_name == "cityscapes" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=cfg.crop_ratio)
        elif dataset_name == "cocostuff3":
            self.n_classes = 3
            dataset_class = Coco
            extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        elif dataset_name == "cocostuff15":
            self.n_classes = 15
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff27", crop_type=cfg.crop_type, crop_ratio=cfg.crop_ratio)
        elif dataset_name == "cocostuff27" and crop_type is None:
            self.n_classes = 27
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if image_set == "val":
                extra_args["subset"] = 7
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
        feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            model_type, nice_dataset_name, image_set, crop_type, cfg.res))
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file) or compute_knns:
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

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
        coord_entries = torch.meshgrid([torch.linspace(-1, 1, pack[0].shape[1]),
                                        torch.linspace(-1, 1, pack[0].shape[2])])
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
