from setuptools import find_packages
from distutils.core import setup

setup(
    name="stego",
    version="0.0.1",
    author="Piotr Libera, Jonas Frey, Matias Mattamala",
    author_email="plibera@student.ethz.ch, jonfrey@ethz.ch, matias@leggedrobotics.com",
    packages=find_packages(),
    python_requires=">=3.6",
    description="Self-supervised semantic segmentation package based on the STEGO model",
    install_requires=[],
)
