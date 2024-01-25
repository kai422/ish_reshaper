from setuptools import setup, find_packages

# Code Based on DropIT: Dropping Intermediate Tensors for Memory-Efficient DNN Training (ICLR'23). https://github.com/chenjoya/dropit

setup(
    name='ishreshaper',
    packages=find_packages(where=("ishreshaper")),
    version='0.1.0',
    install_requires=[
        "torch>=1.12.1",
        "torchvision>=0.13.1"
    ]
)