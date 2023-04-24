import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchmetrics import Metric
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from PIL import Image
import sys
import random
from torch.utils.data import DataLoader, Dataset
import os



class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = root
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


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