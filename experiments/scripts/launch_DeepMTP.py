import sys
sys.path.append("../../")
sys.path.append("../datasets/data_utils/")


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from IMPACT import utils
utils.set_seed(0)
sys.path.insert(0, str((Path.cwd() / "../datasets/data_utils").resolve()))

import json
import torch

import logging

import DeepMTP_util

def parse_args():
    parser = argparse.ArgumentParser(
        description="A program that runs the CAT testing session"
    )
    parser.add_argument('dataset_name', help="the dataset name")
    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    args = parse_args()

    utils.setuplogger(verbose = True, log_name="NMF_cornac")

    # choose dataset here
    dataset_name = args.dataset_name
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
    utils.set_seed(config['seed'])
    dataset_name += version


    seed = 0
    utils.set_seed(0)
    
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
            logging.info(dataset_name)
            config['learning_rate'] = 0.00833
            config['lambda'] = 1e-7
        case "promis":
            logging.info(dataset_name)
            config['learning_rate'] = 0.00027
            config['lambda'] = 3e-7
        case "portrait":
            logging.info(dataset_name)
            config['learning_rate'] = 0.00735
            config['lambda'] = 1e-7
        case "movielens":
            logging.info(dataset_name)
            config['learning_rate'] = 0.00153
            config['lambda'] = 5e-7

    config["dataset_name"] = args.dataset_name 
    
    logging.info(f'#### {dataset_name} ####')
    logging.info(f'#### config : {config} ####')
    
    metrics = DeepMTP_util.test(dataset_name,config)
