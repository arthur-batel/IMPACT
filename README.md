# An interpretable model for multi-target predictions with multi-class outputs

---
This is the official repository of DBPR, a novel interpretable model for Multi-Target Predictions (MTP) with multi-class outputs. The model extends Cognitive Diagnosis Bayesian Personalized Ranking framework to the case of multi-class prediction. The implementatino language is mainly python. It uses pytorch for the implementation of DBPR model itself. This repository contains the four datasets used in the paper, as well jupyter notebooks to re-run the experiments and conduct your own.

## Installing DBPR from source
```bash
git clone https://github.com/arthur-batel/DBPR.git
cd DBPR
make install
conda activate dbpr-env
# open one of the notebooks in the experiments/notebook_examples forlder
```

## Training, testing, evaluating the model and running experiments

All examples of how to train, test, evaluate the model and replicate the experiments are in the `experiments/notebook example` folder.


## Authors

Arthur Batel, Idir Benouaret, Marc Plantevit and Céline Robardet