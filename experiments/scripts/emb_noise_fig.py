from torch.utils import data
import torch

def _evaluate_preds_spec(self, data_loader: data.DataLoader, dim: int):
    # Accumulators for two splits: mask_true (==dim) and mask_false (!=dim)
    acc = {
        True:  {"loss": [], "pred": [], "label": [], "nb_mod": []},
        False: {"loss": [], "pred": [], "label": [], "nb_mod": []},
    }

    def process_mask(mask: torch.Tensor, user_ids, item_ids, labels, dim_ids):
        if not mask.any():
            return
        # align nb_modalities with the *masked* item_ids
        nb_mod_all = data_loader.dataset.nb_modalities[item_ids]
        nb_mod = nb_mod_all[mask]

        u = user_ids[mask]
        i = item_ids[mask]
        d = dim_ids[mask]
        y = labels[mask]

        preds = self.model(u, i, d)
        loss  = self._loss_function(preds, y).float()

        bucket = acc[bool((d == dim).all().item())]  # True if this bucket is ==dim
        bucket["loss"].append(loss)
        bucket["pred"].append(preds.detach())
        bucket["label"].append(y.detach())
        bucket["nb_mod"].append(nb_mod.detach())

    with torch.inference_mode():
        for data_batch in data_loader:
            user_ids = data_batch[:, 0].long()
            item_ids = data_batch[:, 1].long()
            labels   = data_batch[:, 2]
            dim_ids  = data_batch[:, 3].long()

            mask_true  = (dim_ids == dim)
            mask_false = ~mask_true

            # process each split
            process_mask(mask_true,  user_ids, item_ids, labels, dim_ids)
            process_mask(mask_false, user_ids, item_ids, labels, dim_ids)

    def pack(bucket):
        if len(bucket["pred"]) == 0:
            return (torch.tensor([], dtype=torch.float32),
                    torch.tensor([], dtype=torch.float32),
                    torch.tensor([], dtype=torch.float32),
                    torch.tensor([], dtype=torch.long))
        loss_tensor          = torch.stack(bucket["loss"])
        pred_tensor          = torch.cat(bucket["pred"],  dim=0)
        label_tensor         = torch.cat(bucket["label"], dim=0)
        nb_modalities_tensor = torch.cat(bucket["nb_mod"], dim=0)
        return loss_tensor, pred_tensor, label_tensor, nb_modalities_tensor

    return (*pack(acc[True]), *pack(acc[False]))


def evaluate_emb_qual(self, test_dataset: data.Dataset, dim: int, noise_level: float):
    """
    Temporarily add multiplicative Gaussian noise to users_emb[:, dim],
    evaluate on items with dim_ids == dim and != dim separately, and restore.
    """
    # Cache original column
    with torch.no_grad():
        original_col = self.model.users_emb.weight[:, dim].detach().clone()

    try:
        # Add noise based on the *original* column so runs don't accumulate noise
        with torch.no_grad():
            col   = original_col
            std_c = col.std(unbiased=False)
            noise = torch.randn_like(col) * (noise_level * std_c)
            self.model.users_emb.weight[:, dim].copy_(col + noise)

        # Build loader and evaluate
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=100_000,
            shuffle=False,
            pin_memory=self.config['pin_memory'],
            num_workers=self.config['num_workers'],
        )

        (loss_t, pred_t, label_t, nbmod_t,
         loss_f, pred_f, label_f, nbmod_f) = _evaluate_preds_spec(self, test_loader, dim)

        # Prepare metric inputs
        def to_metric_types(pred, label, nbmod):
            return pred.to(torch.double), label.to(torch.double), nbmod.to(torch.long)

        results_true, results_false = {}, {}

        if pred_t.numel() > 0:
            p, y, n = to_metric_types(pred_t, label_t, nbmod_t)
            results_true = {
                metric: self.pred_metric_functions[metric](p, y, n).cpu().item()
                for metric in ['rmse', 'mae']
            }
        else:
            results_true = {metric: None for metric in ['rmse', 'mae']}

        if pred_f.numel() > 0:
            p, y, n = to_metric_types(pred_f, label_f, nbmod_f)
            results_false = {
                metric: self.pred_metric_functions[metric](p, y, n).cpu().item()
                for metric in ['rmse', 'mae']
            }
        else:
            results_false = {metric: None for metric in ['rmse', 'mae']}

    finally:
        # Always restore
        with torch.no_grad():
            self.model.users_emb.weight[:, dim].copy_(original_col)

    return results_true, results_false
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="A program that runs the CAT testing session"
    )
    parser.add_argument('dataset', help="the dataset name")
    parser.add_argument('num_response', type=int)
    parser.add_argument('--i_fold', type=int)
    parser.add_argument('--seed', type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils import data
    import torch
    import warnings

    from matplotlib.pyplot import subplots
    import json
    from IMPACT import utils
    from IMPACT import model
    from IMPACT import dataset
    import torch
    import pandas as pd
    
    import numpy as np
    from importlib import reload
    import matplotlib.pyplot as plt
    import logging
    
    # --- config ---
    noise_steps = 10
    seed = int(args.seed)
    i_fold = int(args.i_fold)
    dataset_name=args.dataset
    num_responses=int(args.num_response)
    
    x = np.arange(noise_steps) / noise_steps  # noise levels
    
    n_dims = None
    ratios_rmse = None
    ratios_mae  = None
    rmse = None
    mae = None
    
    config = utils.generate_eval_config(
        i_fold=i_fold,
        seed=seed,
        save_params=False,
        dataset_name=dataset_name,
        learning_rate=0.02026,
        lambda_=1.2e-5,
        batch_size=2048,
        num_epochs=2,
        valid_metric='rmse',
        pred_metrics=['rmse', 'mae'],
        load_params=True
    )
    config['num_responses'] = num_responses
    
    train_data, valid_data, test_data = utils.prepare_dataset(config, i_fold=i_fold)

    # initialize dim count and storage once
    if ratios_rmse is None:
        n_dims = train_data.n_categories
        ratios_rmse = np.full((n_dims, noise_steps), np.nan, dtype=float)
        ratios_mae  = np.full((n_dims, noise_steps), np.nan, dtype=float)
        moving_mae = np.full((n_dims, noise_steps), np.nan, dtype=float)

    algo = model.IMPACT(**config)
    algo.init_model(train_data, valid_data)

    # Evaluate each dimension
    for dim in range(n_dims):
        for n in range(noise_steps):
            noise_level = n / noise_steps
            r_dim, r_other = evaluate_emb_qual(algo, test_data, dim, noise_level)

            rmse_dim   = r_dim.get('rmse', None)
            rmse_other = r_other.get('rmse', None)
            mae_dim    = r_dim.get('mae', None)
            mae_other  = r_other.get('mae', None)

            # RMSE ratio
            if (rmse_dim is not None) and (rmse_other not in [None, 0]):
                ratios_rmse[dim, n] = rmse_dim / rmse_other

            # MAE ratio
            if (mae_dim is not None) and (mae_other not in [None, 0]):
                ratios_mae[dim, n] = mae_dim / mae_other

            if (mae_dim not in [None, 0]):
                moving_mae[dim,n] = mae_dim

    print("MAE dim", moving_mae)
    print("MAE ratio",ratios_mae)            
