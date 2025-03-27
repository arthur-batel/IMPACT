from setuptools import setup, find_packages

setup(
    name='lirisimpact',
    version='2.0.13',
    author='Arthur BATEL',
    author_email='arthur.batel@insa-lyon.fr',
    packages=find_packages(),
    description="""IMPACT framework, an interpretable multi-target framework for multi-class outputs""",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    url='https://github.com/arthur-batel/IMPACT.git',
    install_requires=[
        'torch',
        'vegas',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'numba',
        'tensorboardX',
    ],  # And any other dependencies foo needs
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
