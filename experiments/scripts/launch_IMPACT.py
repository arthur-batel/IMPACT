from IMPACT.utils import prepare_dataset, set_seed, setuplogger, generate_eval_config
from IMPACT.model import IMPACT
from datetime import datetime
import logging
import argparse
import pandas as pd
import numpy as np

def main(config : dict, i_fold:int) :

    logging.info(f'#### {config["dataset_name"]} ####')
    logging.info(f'#### config : {config} ####')
    config['embs_path']='../embs/'+str(config["dataset_name"])
    config['params_path']='../ckpt/'+str(config["dataset_name"])

    pred_metrics = {m:[] for m in config['pred_metrics']}
    profile_metrics = {m:[] for m in config['profile_metrics']}

    #gc.collect()
    #torch.cuda.empty_cache()

    # Dataset downloading for doa and rm
    #warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    #warnings.filterwarnings("ignore", category=RuntimeWarning)

    train_data, valid_data, test_data = prepare_dataset(config, i_fold=i_fold)

    algo = IMPACT(**config)

    # Init model
    algo.init_model(train_data, valid_data)

    # train model ----
    algo.train(train_data, valid_data)

    # test model ----

    eval_m = algo.evaluate_predictions(test_data)
    for m in pred_metrics.keys():
        logging.info(f'{m} : {eval_m[m]}')

    np.save('../preds/preds.npy',eval_m['preds'].detach().cpu().numpy())
    np.save('../preds/labels.npy',eval_m['labels'].detach().cpu().numpy())
    np.save('../preds/nb_modalities.npy',eval_m['nb_modalities'].detach().cpu().numpy())

    emb = algo.model.users_emb.weight.detach().cpu().numpy()
    eval_m = algo.evaluate_profiles(test_data)
    for m in profile_metrics.keys():
        logging.info(f'{m} : {eval_m[m]}')
        
    pd.DataFrame(emb).to_csv("../embs/"+config["dataset_name"]+"_IMPACT_cornac_Iter_fold"+str(i_fold)+"_seed_"+str(config["seed"])+".csv",index=False,header=False)



def parse_args():
    parser = argparse.ArgumentParser(
        description="A program that runs the CAT testing session"
    )
    parser.add_argument('dataset_name', help="the dataset name")
    parser.add_argument(
        '--i_fold',
        type=int,
        default=None,
        help="0-indexed fold number (if omitted runs all folds)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    now = datetime.now()
    setuplogger(verbose = True, log_name=f"IMPACT_{args.dataset_name}_{args.i_fold}")
    config = generate_eval_config(esc = 'error', valid_metric= 'rmse', pred_metrics = ['rmse', 'mae','mi_acc','mi_acc_w1','mi_acc_w2' ], profile_metrics = ['doa', 'pc-er', 'rm'], save_params=True, seed=args.seed, i_fold=args.i_fold)
    set_seed(config["seed"])

    match args.dataset_name : 
        case  "postcovid":
            config['learning_rate'] = 0.1
            config['lambda'] = 2e-7
            config['d_in'] = 9
            config['num_responses'] = 15    
        case "movielens":
            config['learning_rate'] = 0.01
            config['lambda'] = 2e-7
            config['d_in'] = 10
            config['num_responses'] = 12
        case "portrait":
            config['learning_rate'] = 0.04568
            config['lambda'] = 2e-7
            config['d_in'] = 6
            config['num_responses'] = 12
        case "promis":
            config['learning_rate'] = 0.01227
            config['lambda'] = 1e-7
            config['d_in'] = 6
            config['num_responses'] = 13

    config["dataset_name"] = args.dataset_name 
    logging.info(config["dataset_name"])
    main(config,args.i_fold)
    


