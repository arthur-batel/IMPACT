import json
import numpy as np
import torch
import random
import pandas as pd
import numpy as np
from math import pow
import warnings
import torch
import logging
import sys
from datetime import datetime
from numba import jit
import numba
from IMPACT import dataset


def setuplogger(verbose: bool = True, log_path: str = "../../experiments/logs/", log_name: str = None):
    root = logging.getLogger()
    if verbose:
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.ERROR)

    # Stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        stream_handler.setLevel(logging.INFO)
    else:
        stream_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    formatter.default_time_format = "%M:%S"
    formatter.default_msec_format = ""
    stream_handler.setFormatter(formatter)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    if log_name is not None:
        now = datetime.now()
        time_str = now.strftime("_%d:%m:%y_%S:%M")
        file_handler = logging.FileHandler(log_path + log_name + time_str + ".log")

        if verbose:
            file_handler.setLevel(logging.INFO)
        else:
            file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Add new handlers
    root.addHandler(stream_handler)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("CUDA is not available. Skipping CUDA seed setting.")


@numba.njit
def compute_cov(x, y):
    # Compute sample covariance (same as np.cov(x,y)[0,1]) with denominator (N-1)
    n = x.size
    if n < 2:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    s = 0.0
    for i in range(n):
        s += (x[i] - mx) * (y[i] - my)
    return s / (n - 1)


@numba.njit
def reverse_array(arr):
    n = arr.size
    res = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        res[i] = arr[n - 1 - i]
    return res


@numba.njit
def compute_rm_fold(emb, d, concept_array, concept_lens):
    # emb: (n_users, n_dims)
    # d: structured array with fields (student_id, item_id, correct, concept_id)
    # concept_array, concept_lens: from preprocess

    n_users, n_dims = emb.shape
    U_resp_sum = np.zeros((n_users, n_dims), dtype=np.float64)
    U_resp_nb = np.zeros((n_users, n_dims), dtype=np.float64)

    # Fill U_resp_sum and U_resp_nb
    for rec in d:
        student_id = rec[0]
        item_id = rec[1]
        correct_val = rec[2]
        length = concept_lens[item_id]
        for i in range(length):
            cid = concept_array[item_id, i]
            U_resp_sum[student_id, cid] += correct_val
            U_resp_nb[student_id, cid] += 1.0

    # Compute U_ave = U_resp_sum / U_resp_nb where nb>0 else 0
    U_ave = np.zeros((n_users, n_dims), dtype=np.float64)
    for i in range(n_users):
        for j in range(n_dims):
            if U_resp_nb[i, j] > 0:
                U_ave[i, j] = U_resp_sum[i, j] / U_resp_nb[i, j]
            else:
                U_ave[i, j] = 0.0

    # Build u_array and e_array
    # Condition: user answered any question => at least one dim in U_ave[i_user] != 0.0
    # Also handle NaNs: If any appear, handle them manually
    u_list = []
    e_list = []
    for i_user in range(n_users):
        # Check if user answered any question
        answered_any = False
        for j in range(n_dims):
            val = U_ave[i_user, j]
            # NaN check
            if val != val:  # val != val means val is NaN
                # Replace NaN with 0
                U_ave[i_user, j] = 0.0
            if U_resp_nb[i_user, j] > 0:
                answered_any = True

        if answered_any:
            # Create copies to avoid modifying emb/U_ave arrays directly
            u_vec = U_ave[i_user].copy()
            e_vec = emb[i_user].copy()

            # If a dimension not answered => it's already 0.0 in U_ave
            # Set corresponding dimension in e_vec to 0 if not answered:
            # Actually we already know unanswered are 0 in U_ave. We'll do same for e_vec:
            for j in range(n_dims):
                if U_resp_nb[i_user, j] == 0:
                    e_vec[j] = 0.0

            u_list.append(u_vec)
            e_list.append(e_vec)

    if len(u_list) == 0:
        # No users answered anything
        return np.nan

    # Convert lists to arrays
    # Numba cannot directly convert list of arrays if their shape is known.
    # But here each element should have the same shape: (n_dims,)
    # We'll allocate arrays directly:
    n_users_filtered = len(u_list)
    u_array = np.zeros((n_users_filtered, n_dims), dtype=np.float64)
    e_array = np.zeros((n_users_filtered, n_dims), dtype=np.float64)
    for i in range(n_users_filtered):
        for j in range(n_dims):
            u_array[i, j] = u_list[i][j]
            e_array[i, j] = e_list[i][j]

    c = 0.0
    s = 0

    # For each dimension:
    # We must extract the users who answered this dimension (u_array[i,dim] !=0)
    # Then compute covariance and do sorting logic
    for dim in range(n_dims):
        # Count how many users answered this dim
        count_true = 0
        for i_user in range(n_users_filtered):
            if u_array[i_user, dim] != 0.0:
                count_true += 1

        if count_true == 0:
            continue

        # Extract those users' responses
        X_u = np.empty(count_true, dtype=np.float64)
        X_e = np.empty(count_true, dtype=np.float64)
        idx_pos = 0
        for i_user in range(n_users_filtered):
            if u_array[i_user, dim] != 0.0:
                X_u[idx_pos] = u_array[i_user, dim]
                X_e[idx_pos] = e_array[i_user, dim]
                idx_pos += 1

        # Compute covariance
        cov_val = compute_cov(X_u, X_e)
        if np.isnan(cov_val):
            continue

        if cov_val > 0:
            # Sort both arrays
            X_u_sorted = np.sort(X_u)
            X_e_sorted = np.sort(X_e)
            cov_star = compute_cov(X_u_sorted, X_e_sorted)
            if np.isnan(cov_star) or cov_star == 0:
                # If cov_star is zero or NaN, handle gracefully
                continue
            rm = cov_val / cov_star
        elif cov_val == 0:
            rm = 0.0
        else:
            # cov_val < 0
            X_u_sorted = np.sort(X_u)
            X_e_sorted = np.sort(X_e)
            X_e_reversed = reverse_array(X_e_sorted)
            cov_prime = compute_cov(X_u_sorted, X_e_reversed)
            if np.isnan(cov_prime) or cov_prime == 0:
                continue
            rm = -cov_val / cov_prime

        c += rm
        s += 1

    if s == 0:
        return np.nan
    return c / s


def compute_rm(embs: list, dataset_name: str, seed: int = 0, fold_nb: int = 5):
    # Filter warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Load concept_map and preprocess
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    concept_array, concept_lens = preprocess_concept_map(concept_map)

    set_seed(seed)
    nb_folds = fold_nb
    mean_c = []

    for i_fold in range(nb_folds):
        emb = embs[i_fold]
        if not isinstance(emb, np.ndarray):
            emb = emb.detach().cpu().numpy().astype(np.float64)
        else:
            emb = emb.astype(np.float64)

        d = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
            encoding='utf-8',
            dtype={'student_id': int, 'item_id': int, "correct": float, "concept_id": int}
        ).to_records(index=False)

        # Ensure the d structured array fields have correct types
        # Assume d has fields (student_id, item_id, correct, concept_id)
        # Just ensure types are compatible:
        # They should already be int, int, float, int from dtype specification.
        # If needed: d = d.astype([('student_id', np.int32), ('item_id', np.int32),
        #                          ('correct', np.float64), ('concept_id', np.int32)])
        # But likely no need if dtype in read_csv is correct.

        c = compute_rm_fold(emb, d, concept_array, concept_lens)
        mean_c.append(c)

    logging.info("rm : " + str(np.mean(mean_c)) + "+- " + str(np.std(mean_c)))


def preprocess_concept_map(concept_map):
    max_item_id = max(concept_map.keys())  # Find the max item_id to define array size
    max_concept_length = max(len(v) for v in concept_map.values())

    # Create an array with the dimensions: (max_item_id+1, max_concept_length)
    concept_map_array = np.zeros((max_item_id + 1, max_concept_length), dtype=np.int64)

    for k, v in concept_map.items():
        concept_map_array[k, :len(v)] = v

    return concept_map_array


def compute_ave_resp(user_n, dim_n, d, concept_map):
    U_resp_sum = torch.zeros(size=(user_n, dim_n))
    U_resp_nb = torch.zeros(size=(user_n, dim_n))

    for l in d:
        U_resp_sum[l[0], concept_map[l[1]]] += l[2]
        U_resp_nb[l[0], concept_map[l[1]]] += 1

    return U_resp_sum / U_resp_nb


def compute_corr_coeff(embs: list, dataset_name: str, seed: int = 0, fold_nb: int = 5):
    # Filter out the specific warning
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    set_seed(seed)

    nb_folds = fold_nb
    mean_c = []
    for i_fold in range(nb_folds):
        #print("-- i fold " + str(i_fold) + " --")
        emb = embs[i_fold]
        d = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int,
                                                                                 "correct": float, "concept_id": int})

        U_resp_sum = torch.zeros(size=(emb.shape[0], emb.shape[1]))
        U_resp_nb = torch.zeros(size=(emb.shape[0], emb.shape[1]))

        for l in d:
            U_resp_sum[l[0], concept_map[l[1]]] += l[2]
            U_resp_nb[l[0], concept_map[l[1]]] += 1

        U_ave = U_resp_sum / U_resp_nb

        # list with all the dims and zeros when the users didn't gave any answers
        u_list_1 = []
        e_list_1 = []

        for i_user, u in enumerate(U_ave):
            mask = ~u.isnan()  # mask of the dim the user gave responses to
            if mask.any():  # if the user answered any qurstions in the test set
                v = u
                v[~mask] = 0
                e = emb[i_user]
                e[~mask] = 0

                u_list_1.append(v)
                e_list_1.append(e)

        u_array = np.array(u_list_1)
        e_array = np.array(e_list_1)

        c = 0
        s = 0
        for dim in range(emb.shape[1]):
            mask = np.nonzero(u_array[:, dim])  # mask of the dims users gave answers to in the test set
            X = np.array([u_array[mask, dim],
                          e_array[mask, dim]]).squeeze(1)

            corr = np.corrcoef(X)[0][1]
            if not np.isnan(corr):
                c += np.corrcoef(X)[0][1]
                s += 1
        c /= s
        mean_c.append(c)
    logging.info("pc-er : " + str(np.mean(mean_c)) + "+- " + str(np.std(mean_c)))


def preprocess_concept_map(concept_map):
    # concept_map: dict[item_id -> list of concept_ids]
    items = sorted(concept_map.keys())
    max_len = max(len(v) for v in concept_map.values())

    concept_array = np.full((max(items) + 1, max_len), -1, dtype=np.int32)
    concept_lens = np.zeros(max(items) + 1, dtype=np.int32)

    for k, v in concept_map.items():
        v = np.array(v, dtype=np.int32)
        concept_array[k, :len(v)] = v
        concept_lens[k] = len(v)

    return concept_array, concept_lens


def compute_corr_coeff_fold(emb, dataset_name: str, seed: int = 0, i_fold: int = 0):
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Load and prepare concept_map
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    concept_array, concept_lens = preprocess_concept_map(concept_map)

    set_seed(seed)

    d = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                    encoding='utf-8', dtype={'student_id': int, 'item_id': int,
                                             "correct": float, "concept_id": int})
    # Convert d to a structured numpy array
    d = d.to_records(index=False)

    # Ensure emb is a numpy array
    # If emb is a torch tensor, do: emb = emb.detach().cpu().numpy()
    # Here we assume emb is already a numpy array
    if not isinstance(emb, np.ndarray):
        emb = emb.cpu().detach().numpy()

    return corr_coeff(emb, d, concept_array, concept_lens)


@numba.njit
def corr_coeff(emb, d, concept_array, concept_lens):
    n_users, n_dims = emb.shape
    U_resp_sum = np.zeros((n_users, n_dims), dtype=np.float64)
    U_resp_nb = np.zeros((n_users, n_dims), dtype=np.float64)

    # Fill in U_resp_sum and U_resp_nb
    for rec in d:
        student_id = rec[0]
        item_id = rec[1]
        correct_val = rec[2]
        length = concept_lens[item_id]
        for i in range(length):
            cid = concept_array[item_id, i]
            U_resp_sum[student_id, cid] += correct_val
            U_resp_nb[student_id, cid] += 1.0

    U_ave = np.zeros((n_users, n_dims), dtype=np.float64)
    valid_mask = U_resp_nb > 0

    # Compute averages without fancy indexing
    for i in range(n_users):
        for j in range(n_dims):
            if valid_mask[i, j]:
                U_ave[i, j] = U_resp_sum[i, j] / U_resp_nb[i, j]
            else:
                U_ave[i, j] = 0.0

    # Count how many users answered at least one dimension
    count_users = 0
    for i_user in range(n_users):
        answered_any = False
        for x in range(n_dims):
            if valid_mask[i_user, x]:
                answered_any = True
                break
        if answered_any:
            count_users += 1

    if count_users == 0:
        # No user answered anything
        return np.nan

    # Pre-allocate arrays
    u_array = np.empty((count_users, n_dims), dtype=np.float64)
    e_array = np.empty((count_users, n_dims), dtype=np.float64)

    # Fill u_array and e_array
    idx_user = 0
    for i_user in range(n_users):
        answered_any = False
        for x in range(n_dims):
            if valid_mask[i_user, x]:
                answered_any = True
                break
        if answered_any:
            for j in range(n_dims):
                u_array[idx_user, j] = U_ave[i_user, j]
                e_array[idx_user, j] = emb[i_user, j]
            idx_user += 1

    c = 0.0
    s = 0

    for dim in range(n_dims):
        # Count how many users answered this dimension
        count_true = 0
        for i_user in range(u_array.shape[0]):
            if u_array[i_user, dim] != 0.0:
                count_true += 1

        if count_true == 0:
            continue

        X_u = np.empty(count_true, dtype=np.float64)
        X_e = np.empty(count_true, dtype=np.float64)
        idx_pos = 0
        for i_user in range(u_array.shape[0]):
            if u_array[i_user, dim] != 0.0:
                X_u[idx_pos] = u_array[i_user, dim]
                X_e[idx_pos] = e_array[i_user, dim]
                idx_pos += 1

        corr = pearsonr(X_u, X_e)
        if not np.isnan(corr):
            c += corr
            s += 1

    if s == 0:
        return np.nan
    return c / s


@numba.njit
def pearsonr(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    xm = x - mx
    ym = y - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if r_den == 0.0:
        return np.nan
    return r_num / r_den


# Helper function to convert list of lists into padded NumPy array
def _preprocess_concept_map(list_concept_map, max_len):
    concept_map_array = -np.ones((len(list_concept_map), max_len), dtype=np.int64)  # Initialize with -1
    for i, concepts in enumerate(list_concept_map):
        concept_map_array[i, :len(concepts)] = concepts  # Copy valid values into array
    return concept_map_array


# Example: Converting list_q to NumPy array with consistent size (padded if necessary)
def _preprocess_list_q(list_q, max_len):
    q_array = -np.ones((len(list_q), max_len), dtype=np.int64)  # Initialize with -1 for padding
    for i, q_i in enumerate(list_q):
        q_array[i, :len(q_i)] = q_i  # Copy each list q_i into the array, pad with -1 if shorter
    return q_array


def compute_doa(embs: list, dataset_name: str, algo_name: str):
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))

    doas = []
    for i_fold in range(5):
        train_quadruplets = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
            encoding='utf-8').to_records(index=False,
                                         column_dtypes={'student_id': int, 'item_id': int, "dimension_id": int,
                                                        "correct": float, "concept_id": int})
        train_data = dataset.LoaderDataset(train_quadruplets, concept_map, metadata)

        R = train_data.log_tensor.numpy()
        E = embs[i_fold]

        doas.append(evaluate_doa(E, R, metadata, concept_map))

    logging.info("doa : " + str(np.mean(doas)) + "+- " + str(np.std(doas)))


def evaluate_doa(E, R, metadata, concept_map):
    q = {}
    for r in range(metadata['num_item_id']):
        q[r] = []

    for u, i in torch.tensor(R).nonzero():
        q[i.item()].append(u.item())

    max_concepts_per_item = 0
    list_concept_map = []
    for d in concept_map:
        list_concept_map.append(concept_map[d])
        l = len(concept_map[d])
        if l > max_concepts_per_item:
            max_concepts_per_item = l

    list_q = []
    list_q_len = []
    for key in q.keys():
        list_q.append(q[key])
        list_q_len.append(len(q[key]))

    max_q_len = max(len(q_i) for q_i in list_q)
    q_array = _preprocess_list_q(list_q, max_q_len)
    concept_map_array = _preprocess_concept_map(list_concept_map, max_concepts_per_item)

    # Convert q_len to a NumPy array
    q_len = np.array(list_q_len, dtype=np.int32)

    num_dim = metadata['num_dimension_id']
    num_user = metadata['num_user_id']

    # Optionally ensure concept indices are in range inside _compute_doa:
    # You can either filter concept_indices there or ensure _preprocess_concept_map
    # doesn't produce out-of-range indices.

    return _compute_doa(q_array, q_len, num_dim, E, concept_map_array, R, num_user)


@numba.jit(nopython=True, cache=True)
def _compute_doa(q, q_len, num_dim, E, concept_map_array, R, num_user):
    s = np.zeros(shape=(1, num_dim))
    beta = np.zeros(shape=(1, num_dim))

    for i in range(len(q)):  # Adjusted to loop over indices
        concept_indices = concept_map_array[i]
        concept_indices = concept_indices[(concept_indices >= 0) & (concept_indices < num_dim)]

        E_i = E[:, concept_indices]  # Index E using NumPy array
        q_i_len = q_len[i]

        for u_i in range(q_i_len - 1):
            u = q[i, u_i]
            for v in q[i, u_i + 1:q_i_len]:
                if R[u, i] > R[v, i]:
                    for idx in range(len(concept_indices)):
                        s[0, concept_indices[idx]] += E_i[u, idx] > E_i[v, idx]
                        beta[0, concept_indices[idx]] += E_i[u, idx] != E_i[v, idx]
                elif R[u, i] < R[v, i]:
                    for idx in range(len(concept_indices)):
                        s[0, concept_indices[idx]] += E_i[u, idx] < E_i[v, idx]
                        beta[0, concept_indices[idx]] += E_i[u, idx] != E_i[v, idx]

    # Avoid division by zero
    for idx in range(num_dim):
        if beta[0, idx] == 0:
            beta[0, idx] = 1

    return s / beta

def _generate_config(dataset_name:str=None, seed: int = 0, load_params: bool = False, save_params: bool = False, embs_path: str = '../embs/',
                    params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error', verbose_early_stopping: str = False, disable_tqdm: bool = True,
                    valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048, num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                    device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False, flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = ['doa'],
                    num_responses: int = 12, low_mem: bool = False) -> dict:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
    return {
        'seed': seed,
        'dataset_name': dataset_name,
        'load_params': load_params,
        'save_params': save_params,
        'embs_path': embs_path,
        'params_path': params_path,
        'early_stopping': early_stopping,
        'esc': esc,
        'verbose_early_stopping': verbose_early_stopping,
        'disable_tqdm': disable_tqdm,
        'valid_metric': valid_metric,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'eval_freq': eval_freq,
        'patience': patience,
        'device': device,
        'lambda': lambda_,
        'tensorboard': tensorboard,
        'flush_freq': flush_freq,
        'pred_metrics': pred_metrics,
        'profile_metrics': profile_metrics,
        'num_responses': num_responses,
        'low_mem': low_mem,
    }

def generate_hs_config(dataset_name:str=None, seed: int = 0, load_params: bool = False, save_params: bool = False, embs_path: str = '../embs/',
                    params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error', verbose_early_stopping: str = False, disable_tqdm: bool = True,
                    valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048, num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                    device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False, flush_freq: bool = True, pred_metrics: list = ['rmse'], profile_metrics: list = [],
                    num_responses: int = 12, low_mem: bool = False) -> dict:
    """
        Generate a configuration dictionary for the model hyperparameter search process.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
            seed (int): Random seed for reproducibility. Default is 0.
            load_params (bool): Whether to load model parameters from a file. Default is False.
            save_params (bool): Whether to save model parameters to a file. Default is False.
            embs_path (str): Path to the directory where embeddings will be saved. Default is '../embs/'.
            params_path (str): Path to the directory where model parameters will be saved. Default is '../ckpt/'.
            early_stopping (bool): Whether to use early stopping during training. Default is True.
            esc (str): Early stopping criterion. Possible values: 'error', 'loss', 'delta_error', 'objectives'. Default is 'error'.
            verbose_early_stopping (str): Whether to print model learning statistics during training (frequency = eval_freq). Default is False.
            disable_tqdm (bool): Whether to disable tqdm progress bars. Default is True.
            valid_metric (str): Metric to be used for hyperparameters selection on the valid dataset (including early stopping). Possible values: 'rmse', 'mae', 'mi_acc'. Default is 'rmse'.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Batch size for training. Default is 2048.
            num_epochs (int): Number of epochs for training. (Maximum number if early stopping) Default is 200.
            eval_freq (int): Frequency of evaluation during training. Default is 1.
            patience (int): Patience for early stopping. Default is 30.
            device (str): Device to be used for training (e.g., 'cpu' or 'cuda'). Default is None.
            lambda_ (float): Regularization parameter. Default is 7.7e-6.
            tensorboard (bool): Whether to use TensorBoard for logging. Default is False.
            flush_freq (bool): Whether to flush the TensorBoard logs frequently. Default is True.
            pred_metrics (list): List of prediction metrics to be used for evaluation. Possible list elements: 'rmse', 'mae', 'r2', 'mi_acc', 'mi_prec', 'mi_rec', 'mi_f1', 'mi_auc' (mi = micro-averaged). Default is ['rmse', 'mae'].
            profile_metrics (list): List of profile metrics to be used for evaluation. Possible list elements: 'doa', 'pc-er', 'rm'. Default is [].
            num_responses (int): Number of responses IMPACT will use for each question in the case of dataset with continuous values. For discrete datasets, num_responses is the MAXIMUM number of responses IMPACT will use for each question. Default is 12.
            low_mem (bool): Whether to enable low memory mode for IMPACT with vector subspaces for question-response embeddings. Default is False.

        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name, seed, load_params, save_params, embs_path, params_path, early_stopping, esc, verbose_early_stopping, disable_tqdm,
                            valid_metric, learning_rate, batch_size, num_epochs, eval_freq, patience, device, lambda_, tensorboard, flush_freq, pred_metrics, profile_metrics,
                            num_responses, low_mem)

def generate_eval_config(dataset_name:str=None, seed: int = 0, load_params: bool = False, save_params: bool = True, embs_path: str = '../embs/' ,
                    params_path: str = '../ckpt/', early_stopping: bool = True, esc: str = 'error', verbose_early_stopping: str = False, disable_tqdm: bool = False,
                    valid_metric: str = 'rmse', learning_rate: float = 0.001, batch_size: int = 2048, num_epochs: int = 200, eval_freq: int = 1, patience: int = 30,
                    device: str = None, lambda_: float = 7.7e-6, tensorboard: bool = False, flush_freq: bool = True, pred_metrics: list = ['rmse', 'mae', 'r2'], profile_metrics: list = ['doa', 'pc-er'],
                    num_responses: int = 12, low_mem: bool = False) -> dict:
    """
        Generate a configuration dictionary for the model evaluation.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
            seed (int): Random seed for reproducibility. Default is 0.
            load_params (bool): Whether to load model parameters from a file. Default is False.
            save_params (bool): Whether to save model parameters to a file. Default is True.
            embs_path (str): Path to the directory where embeddings will be saved. Default is '../embs/'.
            params_path (str): Path to the directory where model parameters will be saved. Default is '../ckpt/'.
            early_stopping (bool): Whether to use early stopping during training. Default is True.
            esc (str): Early stopping criterion. Possible values: 'error', 'loss', 'delta_error', 'objectives'. Default is 'error'.
            verbose_early_stopping (str): Whether to print model learning statistics during training (frequency = eval_freq). Default is False.
            disable_tqdm (bool): Whether to disable tqdm progress bars. Default is False.
            valid_metric (str): Metric to be used for hyperparameters selection on the valid dataset (including early stopping). Possible values: 'rmse', 'mae', 'mi_acc'. Default is 'rmse'.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            batch_size (int): Batch size for training. Default is 2048.
            num_epochs (int): Number of epochs for training. (Maximum number if early stopping) Default is 200.
            eval_freq (int): Frequency of evaluation during training. Default is 1.
            patience (int): Patience for early stopping. Default is 30.
            device (str): Device to be used for training (e.g., 'cpu' or 'cuda'). Default is None.
            lambda_ (float): Regularization parameter. Default is 7.7e-6.
            tensorboard (bool): Whether to use TensorBoard for logging. Default is False.
            flush_freq (bool): Whether to flush the TensorBoard logs frequently. Default is True.
            pred_metrics (list): List of prediction metrics to be used for evaluation. Possible list elements: 'rmse', 'mae', 'r2', 'mi_acc', 'mi_prec', 'mi_rec', 'mi_f1', 'mi_auc' (mi = micro-averaged). Default is ['rmse', 'mae'].
            profile_metrics (list): List of profile metrics to be used for evaluation. Possible list elements: 'doa', 'pc-er', 'rm'. Default is ['doa', 'pc-er'].
            num_responses (int): Number of responses IMPACT will use for each question in the case of dataset with continuous values. For discrete datasets, num_responses is the MAXIMUM number of responses IMPACT will use for each question. Default is 12.
            low_mem (bool): Whether to enable low memory mode for IMPACT with vector subspaces for question-response embeddings. Default is False.

        Returns:
            dict: Configuration dictionary with the specified parameters.
        """
    return _generate_config(dataset_name, seed, load_params, save_params, embs_path, params_path, early_stopping, esc, verbose_early_stopping, disable_tqdm,
                            valid_metric, learning_rate, batch_size, num_epochs, eval_freq, patience, device, lambda_, tensorboard, flush_freq, pred_metrics, profile_metrics,
                            num_responses, low_mem)

def prepare_dataset(config: dict, i_fold:int=0) :
    """
    Prepare the dataset for training, validation, and testing.

    Args:
        config (dict): Configuration dictionary containing dataset name and other parameters.
        i_fold (int): Fold number for cross-validation. Default is 0.

    Returns:
        tuple: A tuple containing:
            - concept_map (dict): A dictionary mapping question IDs to lists of category IDs.
            - train_data (LoaderDataset): Training dataset.
            - valid_data (LoaderDataset): Validation dataset.
            - test_data (LoaderDataset): Testing dataset.
    """
    ## Concept map format : {question_id : [category_id1, category_id2, ...]}
    concept_map = json.load(open(f'../datasets/{config["dataset_name"]}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}

    ## Metadata map format : {"num_user_id": ..., "num_item_id": ..., "num_dimension_id": ...}
    metadata = json.load(open(f'../datasets/{config["dataset_name"]}/metadata.json', 'r'))

    ## Quadruplets format : (user_id, question_id, response, category_id)
    train_quadruplets = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_train_quadruples_vert_{i_fold}.csv',
        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})
    valid_quadruplets = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_valid_quadruples_vert_{i_fold}.csv',
        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})
    test_quadruplets = pd.read_csv(
        f'../datasets/2-preprocessed_data/{config["dataset_name"]}_test_quadruples_vert_{i_fold}.csv',
        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int, "correct": float,
                                                                 "dimension_id": int})

    train_data = dataset.LoaderDataset(train_quadruplets, concept_map, metadata)
    valid_data = dataset.LoaderDataset(valid_quadruplets, concept_map, metadata)
    test_data = dataset.LoaderDataset(test_quadruplets, concept_map, metadata)

    return concept_map, train_data, valid_data, test_data
