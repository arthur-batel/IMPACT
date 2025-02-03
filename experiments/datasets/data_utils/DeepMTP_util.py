import pandas
from DeepMTP.utils.utils import generate_config
from sklearn.metrics import r2_score
import torch
from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.metrics import MAE, RMSE, RatingMetric
import numpy as np
import logging
import warnings
from IMPACT import utils
from IMPACT import dataset
import pandas as pd
import json
from DeepMTP.main import DeepMTP
from scipy.sparse import csr_matrix
from DeepMTP.utils.data_utils import data_process
import gc

def generate_one_hot_matrix(n):
    """
    Returns an n x n sparse matrix where each row i is a one-hot vector
    with a 1 in column i and 0 elsewhere (i.e., the identity matrix in CSR form).
    """
    # Indices for rows and columns (0..n-1)
    row_indices = np.arange(n)
    col_indices = np.arange(n)

    # Data array of ones (each row has exactly one '1')
    data = np.ones(n, dtype=int)

    # Create a sparse CSR matrix of shape (n, n)
    one_hot_csr = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

    return one_hot_csr

def load_dataset(dataset_name : str) :

    gc.collect()
    torch.cuda.empty_cache()

    # read datasets
    i_fold = 0
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k):[int(x) for x in v] for k,v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))
    train_quadruplets = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_train_quadruples_vert_{i_fold}.csv',
                             encoding='utf-8').to_records(index=False,
                                                          column_dtypes={'student_id': int, 'item_id': int,"dimension_id":int,
                                                                         "correct": float,"dimension_id":int})
    valid_quadruplets = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_valid_quadruples_vert_{i_fold}.csv',
                                 encoding='utf-8').to_records(index=False,
                                                              column_dtypes={'student_id': int, 'item_id': int,"dimension_id":int,
                                                                             "correct": float,"dimension_id":int})

    test_quadruplets = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                            encoding='utf-8').to_records(index=False,
                                                         column_dtypes={'student_id': int, 'item_id': int,
                                                                        "correct": float,"dimension_id":int})

    train_data = dataset.LoaderDataset(train_quadruplets, concept_map, metadata)
    valid_data = dataset.LoaderDataset(valid_quadruplets, concept_map, metadata)
    test_data = dataset.LoaderDataset(test_quadruplets, concept_map, metadata)


    logs_train = train_data.raw_data_array[:,:3].cpu().numpy()
    logs_train[:,2] = logs_train[:,2] - 1
    train_df = pandas.DataFrame(logs_train, columns={'instance_id':int,'target_id':int,'value':float})
    train_df = train_df.astype({'instance_id':int,'target_id':int,'value':float})

    logs_valid = valid_data.raw_data_array[:,:3].cpu().numpy()
    logs_valid[:,2] = logs_valid[:,2] - 1
    valid_df = pandas.DataFrame(logs_valid, columns={'instance_id':int,'target_id':int,'value':float})
    valid_df = valid_df.astype({'instance_id':int,'target_id':int,'value':float})

    logs_test = test_data.raw_data_array[:,:3].cpu().numpy()
    logs_test[:,2] = logs_test[:,2] - 1
    test_df = pandas.DataFrame(logs_test, columns={'instance_id':int,'target_id':int,'value':float})
    test_df = test_df.astype({'instance_id':int,'target_id':int,'value':float})
    U = generate_one_hot_matrix(train_data.n_users)
    I = generate_one_hot_matrix(train_data.n_items)
    data = {
        'train' : {'y' : train_df, 'X_instance' : U.toarray()  , 'X_target' : I.toarray() },
        'val' : {'y' : valid_df, 'X_instance' : U.toarray(), 'X_target' : I.toarray()},
        'test' : {'y' : test_df, 'X_instance' : U.toarray(), 'X_target' : I.toarray()},
    }

    return data_process(data, validation_setting='A',verbose = True),metadata


def generate_DeepMTP(config, metadata):
    config_DMTP = generate_config(num_epochs=config['num_epochs'], learning_rate=config['learning_rate'],
                                  decay=config['lambda'], compute_mode=config['device'],
                                  train_batchsize=config['batch_size'], val_batchsize=10000,
                                  patience=config['patience'],
                                  evaluate_train=True, evaluate_val=True, problem_mode="regression",
                                  metrics=['RMSE'], metrics_average=['micro'],
                                  eval_every_n_epochs=config['eval_freq'], use_early_stopping=True,
                                  general_architecture_version='mlp', batch_norm=False, dropout_rate=0,
                                  instance_branch_architecture="MLP", instance_branch_input_dim=metadata["num_user_id"],
                                  instance_branch_params={
                                      "instance_branch_nodes_per_layer": [10, 10, metadata['num_dimension_id']]},
                                  target_branch_params={
                                      "target_branch_nodes_per_layer": [10, 10, metadata['num_dimension_id']]},
                                  target_branch_architecture="MLP", embedding_size=metadata['num_dimension_id'],
                                  comb_mlp_nodes_per_layer = 20,
                                  comb_mlp_layers = 3,
                                  save_model = False,
                                  target_branch_input_dim=metadata["num_item_id"], validation_setting='A')
    return DeepMTP(config_DMTP)


def objective_MTP(trial, config, metadata, generate_model,train,val,test):
    gc.collect()
    torch.cuda.empty_cache()

    lr = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
    lambda_param = trial.suggest_float('lambda', 1e-7, 5e-5, log=True)

    config['learning_rate'] = lr
    config["lambda"] = lambda_param

    model = generate_model(config,metadata)

    # valid model ----
    validation_results = model.train(train, val, test)

    rmse = validation_results['val_RMSE_micro']

    logging.info("-------Trial number : " + str(trial.number) + "\nValues : [" + str(rmse) + "," + "]\nParams : " + str(
        trial.params))

    del model

    gc.collect()
    torch.cuda.empty_cache()

    return rmse


def test(dataset_name: str, config: dict):
    config = config
    # choose dataset here

    config['embs_path'] = '../embs/' + str(dataset_name)
    config['params_path'] = '../ckpt/' + str(dataset_name)

    metrics = {"mae": [], "rmse": [], "pc-er": [], "doa": [], 'rm': []}

    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k): [int(x) for x in v] for k, v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))

    for i_fold in range(3,5):


        train_quadruplets = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_train_quadruples_vert_{i_fold}.csv',
            encoding='utf-8').to_records(index=False,
                                         column_dtypes={'student_id': int, 'item_id': int, "dimension_id": int,
                                                        "correct": float, "dimension_id": int})
        valid_quadruplets = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_valid_quadruples_vert_{i_fold}.csv',
            encoding='utf-8').to_records(index=False,
                                         column_dtypes={'student_id': int, 'item_id': int, "dimension_id": int,
                                                        "correct": float, "dimension_id": int})

        test_quadruplets = pd.read_csv(
            f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
            encoding='utf-8').to_records(index=False,
                                         column_dtypes={'student_id': int, 'item_id': int,
                                                        "correct": float, "dimension_id": int})

        train_data = dataset.LoaderDataset(train_quadruplets, concept_map, metadata)
        valid_data = dataset.LoaderDataset(valid_quadruplets, concept_map, metadata)
        test_data = dataset.LoaderDataset(test_quadruplets, concept_map, metadata)

        logs_train = train_data.raw_data_array[:, :3].cpu().numpy()
        logs_train[:, 2] = logs_train[:, 2] - 1
        train_df = pandas.DataFrame(logs_train, columns={'instance_id': int, 'target_id': int, 'value': float})
        train_df = train_df.astype({'instance_id': int, 'target_id': int, 'value': float})

        logs_valid = valid_data.raw_data_array[:, :3].cpu().numpy()
        logs_valid[:, 2] = logs_valid[:, 2] - 1
        valid_df = pandas.DataFrame(logs_valid, columns={'instance_id': int, 'target_id': int, 'value': float})
        valid_df = valid_df.astype({'instance_id': int, 'target_id': int, 'value': float})

        logs_test = test_data.raw_data_array[:, :3].cpu().numpy()
        logs_test[:, 2] = logs_test[:, 2] - 1
        test_df = pandas.DataFrame(logs_test, columns={'instance_id': int, 'target_id': int, 'value': float})
        test_df = test_df.astype({'instance_id': int, 'target_id': int, 'value': float})
        U = generate_one_hot_matrix(train_data.n_users).toarray()
        U_emb = torch.tensor(U, dtype=torch.float).to('cuda:0')
        I = generate_one_hot_matrix(train_data.n_items).toarray()
        data = {
            'train': {'y': train_df, 'X_instance': U, 'X_target': I},
            'val': {'y': valid_df, 'X_instance': U, 'X_target': I},
            'test': {'y': test_df, 'X_instance': U, 'X_target': I},
        }
        train,val,test,data_info = data_process(data, validation_setting='A', verbose=True)
        #train, val, test = data_process_result
        d = pd.read_csv(f'../datasets/2-preprocessed_data/{dataset_name}_test_quadruples_vert_{i_fold}.csv',
                        encoding='utf-8').to_records(index=False, column_dtypes={'student_id': int, 'item_id': int,
                                                                                 "correct": float, "concept_id": int})
        concept_array, concept_lens = utils.preprocess_concept_map(concept_map)
        train_dataloader = dataset.LoaderDataset(d, concept_map, metadata)

        # Dataset downloading for doa and rm
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for seed in range(3):
            # Set the seed
            utils.set_seed(seed)
            config['seed'] = seed

            config_DMTP = generate_config(num_epochs=config['num_epochs'], learning_rate=config['learning_rate'],
                                          decay=config['lambda'], compute_mode=config['device'],
                                          train_batchsize=config['batch_size'], val_batchsize=10000,
                                          patience=config['patience'],
                                          evaluate_train=True, evaluate_val=True, problem_mode="regression",
                                          metrics=['RMSE','MAE'], metrics_average=['micro'],
                                          eval_every_n_epochs=config['eval_freq'], use_early_stopping=True,
                                          general_architecture_version='mlp', batch_norm=False, dropout_rate=0,
                                          instance_branch_architecture="MLP",
                                          instance_branch_input_dim=metadata["num_user_id"],
                                          instance_branch_params={
                                              "instance_branch_nodes_per_layer": [10, 10,
                                                                                  metadata['num_dimension_id']]},
                                          target_branch_params={
                                              "target_branch_nodes_per_layer": [10, 10, metadata['num_dimension_id']]},
                                          target_branch_architecture="MLP", embedding_size=metadata['num_dimension_id'],
                                          comb_mlp_nodes_per_layer=20,
                                          comb_mlp_layers=3,
                                          save_model=False,
                                          target_branch_input_dim=metadata["num_item_id"], validation_setting='A')

            algo = DeepMTP(config_DMTP)
            validation_results = algo.train(train, val, test)

            # test model ----
            test_results = algo.predict(test, return_predictions=False, verbose=False)[0]
            metrics["mae"].append(test_results['MAE_micro'])
            metrics["rmse"].append(test_results['RMSE_micro'])

            emb = algo.instance_branch_model(U_emb).detach().cpu().numpy()

            metrics["pc-er"].append(utils.corr_coeff(emb, d, concept_array, concept_lens))
            metrics["doa"].append(np.mean(utils.evaluate_doa(emb, train_dataloader.log_tensor.cpu().numpy(), metadata, concept_map)))
            metrics["rm"].append(np.mean(utils.compute_rm_fold(emb, d, concept_array, concept_lens)))

            pd.DataFrame(emb).to_csv(
                "../embs/" + dataset_name + "_GCMC_cornac_Iter_fold" + str(i_fold) + "_seed_" + str(seed) + ".csv",
                index=False, header=False)

    print(metrics)
    df = pd.DataFrame(metrics)
    logging.info(metrics)
    logging.info('rmse : {:.4f} +- {:.4f}'.format(df['rmse'].mean(), df['rmse'].std()))
    logging.info('mae : {:.4f} +- {:.4f}'.format(df['mae'].mean(), df['mae'].std()))
    logging.info('pc-er : {:.4f} +- {:.4f}'.format(df['pc-er'].mean(), df['pc-er'].std()))
    logging.info('doa : {:.4f} +- {:.4f}'.format(df['doa'].mean(), df['doa'].std()))
    logging.info('rm : {:.4f} +- {:.4f}'.format(df['rm'].mean(), df['rm'].std()))

    return metrics
