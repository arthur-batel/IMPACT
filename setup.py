import os
from setuptools import setup, find_packages

SKIP_DEPENDENCIES = os.getenv("SKIP_INSTALL_REQUIRES", "0") == "1"

install_requires = [] if SKIP_DEPENDENCIES else [
    'torch',
    'vegas',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'tqdm',
    'numba',
    'tensorboardX',
],

setup(
    name='lirisimpact',
    version='2.0.25',
    author='Arthur BATEL',
    author_email='arthur.batel@insa-lyon.fr',
    packages=find_packages(),
    description="""IMPACT framework, an interpretable multi-target framework for multi-class outputs""",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    url='https://github.com/arthur-batel/IMPACT.git',
    install_requires=install_requires,# And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.11",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires='>=3.6',
)
