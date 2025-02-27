from setuptools import setup, find_packages

setup(
    name='IMPACT',
    version='2.0.0',
    author='Arthur BATEL',
    author_email='arthur.batel@insa-lyon.fr',
    packages=find_packages(),
    description=""" IMPACT framework for interpretable multi-target prediction for multi-class outputs""",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    url='https://github.com/arthur-batel/IMPACT.git',
    install_requires=[
        'torch',
        'vegas',
        'numpy',
        'scikit-learn',
        'scipy',
        'tqdm',
        'numba',
        'tensorboardX',
    ],  # And any other dependencies foo needs
    entry_points={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
