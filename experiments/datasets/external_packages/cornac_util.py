from sklearn.metrics import r2_score
import torch
from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.metrics import MAE, RMSE, RatingMetric
import numpy as np
import logging
import warnings

from tqdm import tqdm

from IMPACT import utils
from IMPACT import dataset
import pandas as pd
import json
import cornac
import gc


def load_dataset(dataset_name: str,config, i_fold):
    gc.collect()
    torch.cuda.empty_cache()

    # read datasets
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))
    reader = cornac.data.Reader()
    train_data = reader.read(fpath=f'../datasets/2-preprocessed_data/{dataset_name}_train_quadruples_vert_{i_fold}.csv',
                             fmt="UIR", skip_lines=1, sep=",")
    valid_data = reader.read(fpath=f'../datasets/2-preprocessed_data/{dataset_name}_valid_quadruples_vert_{i_fold}.csv',
                             fmt="UIR", skip_lines=1, sep=",")

    test_data = reader.read(fpath=f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                             fmt="UIR", skip_lines=1, sep=",")

    eval_method = cornac.eval_methods.BaseMethod.from_splits(
        train_data=train_data, test_data=test_data, val_data=valid_data, rating_threshold=1.5, exclude_unknowns=False,
        verbose=False, seed=config["seed"])

    return eval_method, concept_map, metadata


def objective_GCMC(trial, config, eval_method,generate_model):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    activation_func = trial.suggest_categorical('activation_func', ['sigmoid', 'relu', 'tanh'])

    config['learning_rate'] = lr
    config['activation_func'] = activation_func

    algo = generate_model(config)

    # valid model ----
    test_result, val_result = eval_method.evaluate(model=algo, metrics=[RMSE()], user_based=True)

    rmse = val_result.metric_avg_results['RMSE']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return rmse


def objective_NMF(trial, config,metadata, eval_method,generate_model):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 5e-4, log=True)

    config['learning_rate'] = lr
    config['lambda'] = lambda_param

    algo = generate_model(config, metadata)

    # valid model ----
    test_result, val_result = eval_method.evaluate(model=algo, metrics=[RMSE()], user_based=True)

    rmse = val_result.metric_avg_results['RMSE']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return rmse

def objective_PMF(trial, config,metadata, eval_method,generate_model):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 5e-4, log=True)

    config['learning_rate'] = lr
    config["lambda"] = lambda_param

    algo = generate_model(config, metadata)

    # valid model ----
    test_result, val_result = eval_method.evaluate(model=algo, metrics=[RMSE()], user_based=True)

    rmse = val_result.metric_avg_results['RMSE']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return rmse

def objective_SVD(trial, config,metadata, eval_method,generate_model):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 5e-4, log=True)

    config['learning_rate'] = lr
    config["lambda"] = lambda_param

    algo = generate_model(config, metadata)

    # valid model ----
    test_result, val_result = eval_method.evaluate(model=algo, metrics=[RMSE()], user_based=True)

    rmse = val_result.metric_avg_results['RMSE']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return rmse

def objective_BIVAECF(trial, config,metadata, eval_method,generate_model):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 5e-5, log=True)
    d1 = trial.suggest_int("d1", 10, 40)
    d2 = trial.suggest_int("d2", 10, 40)
    beta_kl = trial.suggest_float('beta_kl', 5e-2, 1, log=True)

    config['learning_rate'] = lr
    config["lambda"] = lambda_param
    config['enc_str'] = [d1,d2]
    config['activation_func'] = 'sigmoid'
    config['beta_kl'] = beta_kl

    algo = generate_model(config, metadata)

    # valid model ----
    test_result, val_result = eval_method.evaluate(model=algo, metrics=[RMSE()], user_based=True)

    rmse = val_result.metric_avg_results['RMSE']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del algo

    gc.collect()
    torch.cuda.empty_cache()

    return rmse

class R2(RatingMetric):

    def __init__(self):
        RatingMetric.__init__(self, name='R2')

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        gt = np.array(gt_ratings)
        pd = np.array(pd_ratings)
        mean = np.mean(gt)
        sst = np.sum(np.square(gt - mean))
        sse = np.sum(np.square(gt - pd))

        r2 = 1 - sse / sst

        return r2

@torch.jit.script
def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

@torch.jit.script
def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor):
    return torch.mean(torch.abs(y_true - y_pred))

@torch.jit.script
def resp_to_mod(responses: torch.Tensor, nb_modalities: torch.Tensor):
    responses = responses - 1  # -> [0,1]
    indexes = torch.round(responses * (nb_modalities - 1)).long()  # -> [0,nb_modalities-1]
    indexes = indexes + 1  # sentinels add -> [1,nb_modalities]
    return indexes


@torch.jit.script
def mod_to_resp(indexes: torch.Tensor, nb_modalities: torch.Tensor):
    indexes = indexes - 1  # sentinels remove -> [0,nb_modalities-1]
    responses = indexes / (nb_modalities - 1)  # -> [0,1]
    responses = responses + 1  # -> [1,2]
    return responses



def test(dataset_name: str, config: dict, generate_algo, find_emb):
    config = config
    # choose dataset here

    config['embs_path'] = '../embs/' + str(dataset_name)
    config['params_path'] = '../ckpt/' + str(dataset_name)

    metrics = {"mae": [], "rmse": [], "r2": [], "pc-er": [], "doa": [], 'rm': [], 'rmse_round' : [], 'mae_round' : []}

    for i_fold in range(5):

        reader = Reader()
        train_data = reader.read(
            fpath=f'../datasets/2-preprocessed_data/{dataset_name}_train_quadruples_vert_{i_fold}.csv', fmt="UIR",
            skip_lines=1, sep=","
            )
        valid_data = reader.read(
            fpath=f'../datasets/2-preprocessed_data/{dataset_name}_valid_quadruples_vert_{i_fold}.csv', fmt="UIR",
            skip_lines=1, sep=","
            )
        test_data = reader.read(
            fpath=f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv', fmt="UIR",
            skip_lines=1, sep=","
            )

        eval_method = BaseMethod.from_splits(
            train_data=train_data, test_data=test_data, val_data=valid_data, rating_threshold=1.5,
            exclude_unknowns=False, verbose=False, seed=config["seed"]
        )

        # homemade dataset for protocol consistency
        eval_method.test_set = cornac_dataset.build(
            data=test_data,
            fmt=eval_method.fmt,
            global_uid_map=eval_method.global_uid_map,
            global_iid_map=eval_method.global_iid_map,
            seed=eval_method.seed,
            exclude_unknowns=eval_method.exclude_unknowns,
        )

        # Dataset downloading for doa and rm
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
        concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
        metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))
        concept_array, concept_lens = utils.preprocess_concept_map(concept_map)
        d = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int,
                                                                                 "correct": float, "concept_id": int})
        train_dataloader = dataset.LoaderDataset(d, concept_map, metadata)
        train_quadruplets = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_train_quadruples_vert_{i_fold}.csv',
            encoding='utf-8').to_records(index=False,
                                         column_dtypes={'student_id': int, 'item_id': int,
                                                        "correct": float, "dimension_id": int})
        train_data = dataset.LoaderDataset(train_quadruplets, concept_map, metadata)
        nb_mod_max = 20
        R_t = train_data.log_tensor.T.to(device='cpu')
        nb_modalities = torch.zeros(R_t.shape[0], dtype=torch.long)
        for item_i, logs in enumerate(R_t):
            unique_logs = torch.unique(logs)
            delta_min = torch.min(
                torch.abs(unique_logs.unsqueeze(0) - unique_logs.unsqueeze(1)) + torch.eye(unique_logs.shape[0]))

            if delta_min < 1 / (nb_mod_max - 1):
                nb_modalities[item_i] = nb_mod_max
            else:
                nb_modalities[item_i] = (torch.round(1 / delta_min) + 1).long()

        for seed in range(3):
            # Set the seed
            utils.set_seed(seed)
            config['seed'] = seed

            algo = generate_algo(config,metadata)

            # test model ----
            test_result, val_result = eval_method.evaluate(model=algo, metrics=[MAE(), RMSE(), R2()], user_based=False)
            metrics["mae"].append(test_result.metric_avg_results['MAE'])
            metrics["rmse"].append(test_result.metric_avg_results['RMSE'])
            metrics["r2"].append(test_result.metric_avg_results['R2'])

            (u_indices, i_indices, r_values) = eval_method.test_set.uir_tuple
            r_preds = np.fromiter(
                tqdm(
                    (
                        algo.rate(user_idx, item_idx).item()
                        for user_idx, item_idx in zip(u_indices, i_indices)
                    ),
                    desc="Rating",
                    disable=False,
                    miniters=100,
                    total=len(u_indices),
                ),
                dtype="float",
            )

            preds = mod_to_resp(resp_to_mod(torch.Tensor(r_preds), nb_modalities[i_indices]),nb_modalities[i_indices])

            metrics["rmse_round"].append(root_mean_squared_error(torch.Tensor(r_values), preds).item())
            metrics["mae_round"].append(mean_absolute_error(torch.Tensor(r_values), preds).item())

            emb =  find_emb(algo)
            emb = emb

            metrics["pc-er"].append(utils.corr_coeff(emb, d, concept_array, concept_lens))
            metrics["doa"].append(
                np.mean(utils.evaluate_doa(emb, train_dataloader.log_tensor.cpu().numpy(), metadata, concept_map)))
            metrics["rm"].append(np.mean(utils.compute_rm_fold(emb, d, concept_array, concept_lens)))

            pd.DataFrame(emb).to_csv(
                "../embs/" + dataset_name + "_GCMC_cornac_Iter_fold" + str(i_fold) + "_seed_" + str(seed) + ".csv",
                index=False, header=False)

    df = pd.DataFrame(metrics)
    logging.info('rmse : {:.4f} +- {:.4f}'.format(df['rmse'].mean(), df['rmse'].std()))
    logging.info('mae : {:.4f} +- {:.4f}'.format(df['mae'].mean(), df['mae'].std()))
    logging.info('r2 : {:.4f} +- {:.4f}'.format(df['r2'].mean(), df['r2'].std()))
    logging.info('pc-er : {:.4f} +- {:.4f}'.format(df['pc-er'].mean(), df['pc-er'].std()))
    logging.info('doa : {:.4f} +- {:.4f}'.format(df['doa'].mean(), df['doa'].std()))
    logging.info('rm : {:.4f} +- {:.4f}'.format(df['rm'].mean(), df['rm'].std()))
    logging.info('rmse_round : {:.4f} +- {:.4f}'.format(df['rmse_round'].mean(), df['rmse_round'].std()))
    logging.info('mae_round : {:.4f} +- {:.4f}'.format(df['mae_round'].mean(), df['mae_round'].std()))

    return metrics

# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import copy
import os
import pickle
import warnings
from collections import Counter, OrderedDict, defaultdict

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix

from cornac.utils import estimate_batches, get_rng, validate_format


class cornac_dataset(object):
    """Training set contains preference matrix

    Parameters
    ----------
    num_users: int, required
        Number of users.

    num_items: int, required
        Number of items.

    uid_map: :obj:`OrderDict`, required
        The dictionary containing mapping from user original ids to mapped integer indices.

    iid_map: :obj:`OrderDict`, required
        The dictionary containing mapping from item original ids to mapped integer indices.

    uir_tuple: tuple, required
        Tuple of 3 numpy arrays (user_indices, item_indices, rating_values).

    timestamps: numpy.array, optional, default: None
        Array of timestamps corresponding to observations in `uir_tuple`.

    seed: int, optional, default: None
        Random seed for reproducing data sampling.

    Attributes
    ----------
    num_ratings: int
        Number of rating observations in the dataset.

    max_rating: float
        Maximum value among the rating observations.

    min_rating: float
        Minimum value among the rating observations.

    global_mean: float
        Average value over the rating observations.

    uir_tuple: tuple
        Tuple three numpy arrays (user_indices, item_indices, rating_values).

    timestamps: numpy.array
        Numpy array of timestamps corresponding to feedback in `uir_tuple`.
        This is only available when input data is in `UIRT` format.
    """

    def __init__(
        self,
        num_users,
        num_items,
        uid_map,
        iid_map,
        uir_tuple,
        timestamps=None,
        seed=None,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.uid_map = uid_map
        self.iid_map = iid_map
        self.uir_tuple = uir_tuple
        self.timestamps = timestamps
        self.seed = seed
        self.rng = get_rng(seed)

        (_, _, r_values) = uir_tuple
        self.num_ratings = len(r_values)
        self.max_rating = np.max(r_values)
        self.min_rating = np.min(r_values)
        self.global_mean = np.mean(r_values)

        self.__user_ids = None
        self.__item_ids = None
        self.__user_data = None
        self.__item_data = None
        self.__chrono_user_data = None
        self.__chrono_item_data = None
        self.__csr_matrix = None
        self.__csc_matrix = None
        self.__dok_matrix = None

        self.ignored_attrs = [
            "__user_ids",
            "__item_ids",
            "__user_data",
            "__item_data",
            "__chrono_user_data",
            "__chrono_item_data",
            "__csr_matrix",
            "__csc_matrix",
            "__dok_matrix",
        ]

    @property
    def user_ids(self):
        """Return the list of raw user ids"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return the list of raw item ids"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

    @property
    def user_data(self):
        """Data organized by user. A dictionary where keys are users,
        values are tuples of two lists (items, ratings) interacted by the corresponding users.
        """
        if self.__user_data is None:
            self.__user_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                u_data = self.__user_data.setdefault(u, ([], []))
                u_data[0].append(i)
                u_data[1].append(r)
        return self.__user_data

    @property
    def item_data(self):
        """Data organized by item. A dictionary where keys are items,
        values are tuples of two lists (users, ratings) interacted with the corresponding items.
        """
        if self.__item_data is None:
            self.__item_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                i_data = self.__item_data.setdefault(i, ([], []))
                i_data[0].append(u)
                i_data[1].append(r)
        return self.__item_data

    @property
    def chrono_user_data(self):
        """Data organized by user sorted chronologically (timestamps required).
        A dictionary where keys are users, values are tuples of three chronologically
        sorted lists (items, ratings, timestamps) interacted by the corresponding users.
        """
        if self.timestamps is None:
            raise ValueError("Timestamps are required but None!")

        if self.__chrono_user_data is None:
            self.__chrono_user_data = defaultdict()
            for u, i, r, t in zip(*self.uir_tuple, self.timestamps):
                u_data = self.__chrono_user_data.setdefault(u, ([], [], []))
                u_data[0].append(i)
                u_data[1].append(r)
                u_data[2].append(t)
            # sorting based on timestamps
            for user, (items, ratings, timestamps) in self.__chrono_user_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_items = [items[i] for i in sorted_idx]
                sorted_ratings = [ratings[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_user_data[user] = (
                    sorted_items,
                    sorted_ratings,
                    sorted_timestamps,
                )
        return self.__chrono_user_data

    @property
    def chrono_item_data(self):
        """Data organized by item sorted chronologically (timestamps required).
        A dictionary where keys are items, values are tuples of three chronologically
        sorted lists (users, ratings, timestamps) interacted with the corresponding items.
        """
        if self.timestamps is None:
            raise ValueError("Timestamps are required but None!")

        if self.__chrono_item_data is None:
            self.__chrono_item_data = defaultdict()
            for u, i, r, t in zip(*self.uir_tuple, self.timestamps):
                i_data = self.__chrono_item_data.setdefault(i, ([], [], []))
                i_data[0].append(u)
                i_data[1].append(r)
                i_data[2].append(t)
            # sorting based on timestamps
            for item, (users, ratings, timestamps) in self.__chrono_item_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_users = [users[i] for i in sorted_idx]
                sorted_ratings = [ratings[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_item_data[item] = (
                    sorted_users,
                    sorted_ratings,
                    sorted_timestamps,
                )
        return self.__chrono_item_data

    @property
    def matrix(self):
        """The user-item interaction matrix in CSR sparse format"""
        return self.csr_matrix

    @property
    def csr_matrix(self):
        """The user-item interaction matrix in CSR sparse format"""
        if self.__csr_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csr_matrix = csr_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.num_users, self.num_items),
            )
        return self.__csr_matrix

    @property
    def csc_matrix(self):
        """The user-item interaction matrix in CSC sparse format"""
        if self.__csc_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csc_matrix = csc_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.num_users, self.num_items),
            )
        return self.__csc_matrix

    @property
    def dok_matrix(self):
        """The user-item interaction matrix in DOK sparse format"""
        if self.__dok_matrix is None:
            self.__dok_matrix = dok_matrix((self.num_users, self.num_items), dtype="float")
            for u, i, r in zip(*self.uir_tuple):
                self.__dok_matrix[u, i] = r
        return self.__dok_matrix

    @classmethod
    def build(
        cls,
        data,
        fmt="UIR",
        global_uid_map=None,
        global_iid_map=None,
        seed=None,
        exclude_unknowns=False,
    ):
        """Constructing Dataset from given data of specific format.

        Parameters
        ----------
        data: array-like, required
            Data in the form of triplets (user, item, rating) for UIR format,
            or quadruplets (user, item, rating, timestamps) for UIRT format.

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        global_uid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_iid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of items.

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        exclude_unknowns: bool, default: False
            Ignore unknown users and items.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        fmt = validate_format(fmt, ["UIR", "UIRT"])

        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()

        uid_map = OrderedDict()
        iid_map = OrderedDict()

        u_indices = []
        i_indices = []
        r_values = []
        valid_idx = []

        ui_set = set()  # avoid duplicate observations
        dup_count = 0

        for idx, (uid, iid, rating, *_) in enumerate(data):
            if exclude_unknowns and (uid not in global_uid_map or iid not in global_iid_map):
                continue

            # if (uid, iid) in ui_set:
            #     dup_count += 1
            #     continue
            ui_set.add((uid, iid))

            uid_map[uid] = global_uid_map.setdefault(uid, len(global_uid_map))
            iid_map[iid] = global_iid_map.setdefault(iid, len(global_iid_map))

            u_indices.append(uid_map[uid])
            i_indices.append(iid_map[iid])
            r_values.append(float(rating))
            valid_idx.append(idx)

        if dup_count > 0:
            warnings.warn("%d duplicated observations are removed!" % dup_count)

        if len(ui_set) == 0:
            raise ValueError("data is empty after being filtered!")

        uir_tuple = (
            np.asarray(u_indices, dtype="int"),
            np.asarray(i_indices, dtype="int"),
            np.asarray(r_values, dtype="float"),
        )

        timestamps = np.fromiter((int(data[i][3]) for i in valid_idx), dtype="int") if fmt == "UIRT" else None

        dataset = cls(
            num_users=len(global_uid_map),
            num_items=len(global_iid_map),
            uid_map=global_uid_map,
            iid_map=global_iid_map,
            uir_tuple=uir_tuple,
            timestamps=timestamps,
            seed=seed,
        )

        return dataset

    @classmethod
    def from_uir(cls, data, seed=None):
        """Constructing Dataset from UIR (User, Item, Rating) triplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 3]
            Data in the form of triplets (user, item, rating)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        return cls.build(data, fmt="UIR", seed=seed)

    @classmethod
    def from_uirt(cls, data, seed=None):
        """Constructing Dataset from UIRT (User, Item, Rating, Timestamp)
        quadruplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 4]
            Data in the form of triplets (user, item, rating, timestamp)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        return cls.build(data, fmt="UIRT", seed=seed)

    def reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        return self

    def num_batches(self, batch_size):
        """Estimate number of batches per epoch"""
        return estimate_batches(len(self.uir_tuple[0]), batch_size)

    def num_user_batches(self, batch_size):
        """Estimate number of batches per epoch iterating over users"""
        return estimate_batches(self.num_users, batch_size)

    def num_item_batches(self, batch_size):
        """Estimate number of batches per epoch iterating over items"""
        return estimate_batches(self.num_items, batch_size)

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        """Create an iterator over batch of indices

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of indices (array of 'int')

        """
        indices = np.arange(idx_range)
        if shuffle:
            self.rng.shuffle(indices)

        n_batches = estimate_batches(len(indices), batch_size)
        for b in range(n_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids

    def uir_iter(self, batch_size=1, shuffle=False, binary=False, num_zeros=0):
        """Create an iterator over data yielding batch of users, items, and rating values

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        binary: bool, optional, default: False
            If `True`, non-zero ratings will be turned into `1`, otherwise, values remain unchanged.

        num_zeros: int, optional, default = 0
            Number of unobserved ratings (zeros) to be added per user. This could be used
            for negative sampling. By default, no values are added.

        Returns
        -------
        iterator : batch of users (array of 'int'), batch of items (array of 'int'),
            batch of ratings (array of 'float')

        """
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            if binary:
                batch_ratings = np.ones_like(batch_items)
            else:
                batch_ratings = self.uir_tuple[2][batch_ids]

            if num_zeros > 0:
                repeated_users = batch_users.repeat(num_zeros)
                neg_items = np.empty_like(repeated_users)
                for i, u in enumerate(repeated_users):
                    j = self.rng.randint(0, self.num_items)
                    while self.dok_matrix[u, j] > 0:
                        j = self.rng.randint(0, self.num_items)
                    neg_items[i] = j
                batch_users = np.concatenate((batch_users, repeated_users))
                batch_items = np.concatenate((batch_items, neg_items))
                batch_ratings = np.concatenate((batch_ratings, np.zeros_like(neg_items)))

            yield batch_users, batch_items, batch_ratings

    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling="uniform"):
        """Create an iterator over data yielding batch of users, positive items, and negative items

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        neg_sampling: str, optional, default: 'uniform'
            How negative item `j` will be sampled. Supported options: {`uniform`, `popularity`}.

        Returns
        -------
        iterator : batch of users (array of 'int'), batch of positive items (array of 'int'),
            batch of negative items (array of 'int')

        """
        if neg_sampling.lower() == "uniform":
            neg_population = np.arange(self.num_items)
        elif neg_sampling.lower() == "popularity":
            neg_population = self.uir_tuple[1]
        else:
            raise ValueError("Unsupported negative sampling option: {}".format(neg_sampling))

        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_pos_items = self.uir_tuple[1][batch_ids]
            batch_pos_ratings = self.uir_tuple[2][batch_ids]
            batch_neg_items = np.empty_like(batch_pos_items)
            for i, (user, pos_rating) in enumerate(zip(batch_users, batch_pos_ratings)):
                neg_item = self.rng.choice(neg_population)
                while self.dok_matrix[user, neg_item] >= pos_rating:
                    neg_item = self.rng.choice(neg_population)
                batch_neg_items[i] = neg_item
            yield batch_users, batch_pos_items, batch_neg_items

    def user_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over user indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of user indices (array of 'int')
        """
        user_indices = np.fromiter(set(self.uir_tuple[0]), dtype="int")
        for batch_ids in self.idx_iter(len(user_indices), batch_size, shuffle):
            yield user_indices[batch_ids]

    def item_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over item indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of item indices (array of 'int')
        """
        item_indices = np.fromiter(set(self.uir_tuple[1]), "int")
        for batch_ids in self.idx_iter(len(item_indices), batch_size, shuffle):
            yield item_indices[batch_ids]

    def add_modalities(self, **kwargs):
        self.user_feature = kwargs.get("user_feature", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.item_text = kwargs.get("item_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.item_image = kwargs.get("item_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in self.ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    def save(self, fpath):
        """Save a dataset to the filesystem.

        Parameters
        ----------
        fpath: str, required
            Path to a file for the dataset to be stored.

        """
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        dataset = copy.deepcopy(self)
        pickle.dump(dataset, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fpath):
        """Load a dataset from the filesystem.

        Parameters
        ----------
        fpath: str, required
            Path to a file where the dataset is stored.

        Returns
        -------
        self : object
        """
        dataset = pickle.load(open(fpath, "rb"))
        dataset.load_from = fpath  # for further loading
        return dataset