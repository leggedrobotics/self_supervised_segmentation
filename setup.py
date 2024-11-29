#
# Copyright (c) 2022-2024, ETH Zurich, Piotr Libera, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
from setuptools import find_packages
from distutils.core import setup

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "tqdm",
    "kornia>=0.6.5",
    "pip",
    "torchvision",
    "torch>=1.21",
    "torchmetrics",
    "pytorch_lightning>=1.6.5",
    "pytest",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pandas",
    "pytictac",
    "torch_geometric",
    "omegaconf",
    "optuna",
    "neptune",
    "fast-slic",
    "hydra-core",
    "prettytable",
    "termcolor",
    "pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git",
    "liegroups@git+https://github.com/mmattamala/liegroups",
    "opencv-python",
    "wget",
    "rospkg",
    "wandb",
    "gdown"
]

setup(
    name="stego",
    version="0.0.1",
    author="Piotr Libera, Jonas Frey, Matias Mattamala",
    author_email="plibera@student.ethz.ch, jonfrey@ethz.ch, matias@leggedrobotics.com",
    packages=find_packages(),
    python_requires=">=3.7",
    description="Self-supervised semantic segmentation package based on the STEGO model",
    install_requires=[INSTALL_REQUIRES],
)
