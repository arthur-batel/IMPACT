# An interpretable model for multi-target predictions with multi-class outputs

---
This is the official repository of IMPACT, a novel interpretable model for Multi-Target Predictions (MTP) with multi-class outputs. The model extends Cognitive Diagnosis Bayesian Personalized Ranking framework to the case of multi-class prediction. The implementatino language is mainly python. It uses pytorch for the implementation of IMPACT model itself. This repository contains the four datasets used in the paper, as well jupyter notebooks to re-run the experiments and conduct your own.

## Installing IMPACT from source
```bash
git clone https://github.com/arthur-batel/IMPACT.git
cd IMPACT
make install
conda activate impact-env
# open one of the notebooks in the experiments/notebook_examples forlder
```

## Requirements
- Linux OS
- conda package manager
- CUDA version >= 12.4
- **pytorch for CUDA** (to install with pip in accordance with your CUDA version : [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/))

## Repository map
- `experiments/` : Contains the jupyter notebooks and datasets to run the experiments of the scientific paper.
    - `experiments/ckpt/` : Folder for models parameter saving
    - `experiments/datasets/` : Contains the raw and pre-processed datasets, as well as there pre-processing jupyter notebook
    - `experiments/embs/` : Folder for user embeddings saving
    - `experiments/hyperparam_search/` : Contains the csv files of the optimal hyperparameter for each method (obtained with Tree-structured Parzen Estimator (TPE) sampler)
    - `experiments/logs/` : Folder for running logs saving
    - `experiments/notebook_example/` : Contains the jupyter notebooks to run the experiments of the scientific paper.
    - `experiments/preds/` : Folder for predictions saving
    - `experiments/tensorboard/` : Folder for tensorboard data saving
- `figs/` : Contains the figures of the paper
- `IMPACT/` : Contains the source code of the IMPACT model
  - `IMPACT/dataset/` : Contains the code of the dataset class
  - `IMPACT/models/` : Contains the code of the **IMPACT model** and its abstract class, handling the learning process
  - `IMPACT/utils/` : Contains utility functions for logging, complex metric computations, configuration handling, etc.
## Authors

Arthur Batel,
arthur.batel@insa-lyon.fr,
INSA Lyon, LIRIS UMR 5205 FR

Marc Plantevit,
marc.plantevit@epita.fr,
EPITA Lyon, EPITA Research Laboratory (LRE) FR

Idir Benouaret,
idir.benouaret@epita.fr,
EPITA Lyon, EPITA Research Laboratory (LRE) FR

Céline Robardet,
celine.robardet@insa-lyon.fr,
INSA Lyon, LIRIS UMR 5205 FR

## Contributor

Lucas Michaëli

