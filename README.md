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

## Prerequisites
- linux OS
- conda package manager

## Training, testing, evaluating the model and running experiments

All examples of how to train, test, evaluate the model and replicate the experiments are in the `experiments/notebook example` folder.


## Authors

Arthur Batel, Idir Benouaret, Marc Plantevit and Céline Robardet
