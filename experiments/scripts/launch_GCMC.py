import sys
sys.path.append("../../")
sys.path.append("../datasets/external_packages/")

import os
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path


sys.path.insert(0, str((Path.cwd() / "../datasets/external_packages").resolve()))

from cornac_util import test
import numpy as np
import random
from datetime import datetime

import json
import torch
import sys

import logging
from cornac.models import GCMC

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
        time_str = now.strftime("_%d:%m:%y_%H:%M:%S")
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


def generate_GCMC(config,metadata = None) :
    return GCMC(
        max_iter=config['num_epochs'],
        learning_rate=config['learning_rate'],
        gcn_out_units=metadata['num_dimension_id'],
        optimizer='adam',
        gcn_agg_accum="sum",
        activation_func=config['activation_func'],
        gcn_agg_units=500,
        train_valid_interval=config['eval_freq'],
        train_early_stopping_patience=config['patience'],
        trainable=True,
        seed=config['seed'],
        verbose=False
    )

def find_emb(algo):
    enc_graph = algo.model.train_enc_graph
    emb, item_out =  algo.model.net.encoder(enc_graph)
    return emb.detach().cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(
        description="A program that runs the CAT testing session"
    )
    parser.add_argument('dataset_name', help="the dataset name")
    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    args = parse_args()

    setuplogger(verbose = True, log_name="GCMC_cornac")
    dataset_name = args.dataset_name
    # choose dataset here

    version= ""#"_small"
    # modify config here
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    config = {
    
        # General params
        'seed' : 0,
    
        # Saving params
        'load_params': False,
        'save_params': False,
        'embs_path' : '../embs/'+str(dataset_name),
        'params_path' :'../ckpt/'+str(dataset_name),
    
        # training mode
        'early_stopping' : True,
        'fast_training' : True, # (Only taken in account if early_stopping == true) If true, doesn't compute valid rmse PC-ER
    
        # Learning params
        'learning_rate': 0.001,
        'batch_size': 2048,
        'num_epochs': 200,
        'num_dim': 10, # for IRT or MIRT todo : is it necessary as we use concepts knowledge number as embedding dimension ?
        'eval_freq' : 1,
        'patience' : 30,
        'device': device,
        'lambda' : 7.7e-6,
        'tensorboard': False,
        'flush_freq' : True,
    
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
        'best_params_path':'',
    
        #For GCCD
        'num_layers': 0,
        'version': 'pair',
        'p_dropout': 0,
        'low_mem_mode' : True,
        'user_nbrs_n' : 10,
        'item_nbrs_n' : 5
    }
    concept_map = json.load(open(f'../datasets/{dataset_name}/concept_map.json', 'r'))
    concept_map = {int(k):[int(x) for x in v] for k,v in concept_map.items()}
    metadata = json.load(open(f'../datasets/{dataset_name}/metadata.json', 'r'))
    set_seed(config['seed'])
    dataset_name += version

    seed = 0
    set_seed(0)
    
    config['seed'] = seed
    config['early_stopping'] = True
    config['esc'] = 'objectives' #'loss' 'delta_objectives'
    config['eval_freq']=1
    config['patience']=30
    
    config['verbose_early_stopping'] = False
    config["tensorboard"] = False
    config['flush_freq'] = False
    config['save_params']= False
    config['disable_tqdm'] = True

    
    match args.dataset_name : 
        case  "postcovid":
            dataset_name = "postcovid"
            config['learning_rate'] = 0.01415
            config['activation_func'] = 'tanh'

        case "movielens":
            logging.info(dataset_name)
            config['learning_rate'] = 0.01892
            config['activation_func'] = 'tanh'
        case "portrait":
            logging.info(dataset_name)
            config['learning_rate'] = 0.01541
            config['activation_func'] = 'tanh'
        case "promis":
            logging.info(dataset_name)
            config['learning_rate'] = 0.02032
            config['activation_func'] = 'relu'

    config["dataset_name"] = args.dataset_name 
    
    logging.info(f'#### {dataset_name} ####')
    logging.info(f'#### config : {config} ####')
    
    metrics = test(dataset_name,config,generate_GCMC,find_emb)