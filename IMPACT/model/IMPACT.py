import functools
import os
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn

import torch.utils.data as data

from IMPACT.model.abstract_model import AbstractModel
from IMPACT.dataset import *
import torch.nn.functional as F

import warnings
import torch

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.*",
    category=UserWarning
)


class CoVWeightingLoss(nn.Module):
    def __init__(self, device: str):
        super().__init__() 
        self.nb_losses: int = 2
        self.device: str = device

        # Initialize tensors for online statistics
        self.t: int = 0  # Time step
        self.mean_L: torch.Tensor = torch.zeros(self.nb_losses, device=self.device)  # Mean of losses
        self.mean_l: torch.Tensor = torch.ones(self.nb_losses, device=self.device)  # Mean of loss ratios
        self.M2: torch.Tensor = torch.zeros(self.nb_losses, device=self.device)  # Sum of squares of differences from the current mean
        self.weights: torch.Tensor = torch.tensor([0.5, 0.5], device=self.device)  # Initial weights

        self.state: str = "train"  # Initialize state (assuming "train" by default)


    @torch.jit.export
    def compute_weights(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        Update the online statistics for each loss.

        Args:
            loss_values (torch.Tensor): Tensor of loss values with shape (nb_losses,).
                                        Should be on the same device as the class.

        Returns:
            torch.Tensor: Updated weights tensor.
        """
        if self.state == "eval":
            return self.weights

        # Detach and clone to prevent unwanted side effects
        L = loss_values.detach()

        # Update counts
        self.t += 1

        if self.t == 1:
            # Initialize mean_L and reset statistics for loss ratios
            self.mean_L = L.clone()
            # Compute M2 based on current weights and mean_l
            # self.M2 = (self.weights * self.mean_l).square()
            self.weights.fill_(0.5)
        else:
            # Update mean_L using Welford's algorithm
            delta = L - self.mean_L
            prev_mean_L = self.mean_L.clone()
            self.mean_L = self.mean_L + delta / self.t

            # Compute loss ratios ℓ_t = L_t / mean_{L_{t-1}}
            # Avoid division by zero by setting ratios to zero where prev_mean_L == 0
            l = torch.where(prev_mean_L != 0, L / prev_mean_L, torch.zeros_like(L))

            # Update loss_ratio_means and loss_ratio_M2 using Welford's algorithm
            d_ratio = l - self.mean_l
            self.mean_l = self.mean_l + d_ratio / self.t

            d2_ratio = l - self.mean_l
            self.M2 = self.M2 + d_ratio * d2_ratio

            # Compute standard deviation
            std = torch.sqrt(self.M2 / (self.t - 1))

            # Compute coefficient of variation c_l = σ_ℓ / mean_ℓ
            # Avoid division by zero by setting c_l to zero where mean_l == 0
            c_l = torch.where(
                self.mean_l != 0,
                std / self.mean_l,
                torch.zeros_like(self.mean_l)
            )

            # Normalize coefficients to get weights α_i
            z = torch.sum(c_l)

            if z > 0:
                self.weights = c_l / z
            else:
                # If sum is zero, assign equal weights
                self.weights.fill_(1.0 / self.nb_losses)

        return self.weights

    def reset(self):
        """
        Reset all statistics to their initial state.
        """
        self.mean_L.zero_()
        self.t = 0
        self.mean_l.zero_()
        self.M2.zero_()

@torch.jit.export
class IMPACTModel(nn.Module):
    '''
    Graph Convolutional Cognitive Diagnostic
    '''

    def __init__(self, user_n: int, item_n: int, concept_n: int, concept_map: dict, train_data: Dataset,
                 valid_data: Dataset, nb_mod_max: int = 12, load_params: bool = False):
        super(IMPACTModel, self).__init__()
        self.user_n: int = user_n
        self.item_n: int = item_n
        self.nb_mod_max: int = nb_mod_max  # Discretized responses: 0.0 to 1.0 in steps of 0.1
        self.nb_mod_max_plus_sent: int = nb_mod_max + 2
        self.concept_n: int = concept_n
        self.concept_map: dict = concept_map

        # Register R as a buffer to ensure it's on the correct device
        self.register_buffer('R', train_data.log_tensor)

        self.device = self.R.device

        # ------ Declare learnable parameters
        ## User embeddings
        self.users_emb = nn.Embedding(user_n, concept_n)
        ## Item-Response embeddings
        self.item_response_embeddings = nn.Embedding(item_n * self.nb_mod_max_plus_sent, concept_n)

        # ------ Initialize Parameters

        ## Initialize users and item-response embeddings
        self.users_emb.weight.data = self.users_emb.weight.data.zero_().to(self.device)
        self.item_response_embeddings.weight.data = self.item_response_embeddings.weight.data.zero_().to(self.device)

        k2q = defaultdict(set)
        for item, concept_list in concept_map.items():
            for concept in concept_list:
                k2q[concept].add(item)
        items_by_concepts = list(map(set, k2q.values()))
        for i, c in enumerate(items_by_concepts):
            c_list = torch.tensor(list(c), dtype=torch.long, device=self.device)
            self.users_emb.weight.data[:, i].add_(
                (self.R[:, c_list].sqrt().sum(dim=1) / (torch.sum(self.R[:, c_list] != 0, dim=1) + 1e-12))
            )

        response_values = torch.linspace(1, 2, steps=self.nb_mod_max_plus_sent, device=self.device)
        for concept_index, items in enumerate(k2q.values()):
            # Convert items to a tensor
            items = torch.tensor(list(items), dtype=torch.long, device=self.device)

            # Compute item-response indices for all responses
            # item_indices shape: [nb_items_in_concept, nb_responses]
            item_indices = items.unsqueeze(1) * self.nb_mod_max_plus_sent + torch.arange(self.nb_mod_max_plus_sent,
                                                                                         device=self.device).unsqueeze(
                0)

            # Flatten to a 1D tensor
            item_indices = item_indices.reshape(-1)  # [nb_items_in_concept * nb_mod_max]

            # Repeat response values for each item
            response_values_repeated = response_values.repeat(len(items)).unsqueeze(
                1)  # [nb_mod*nb_items_in_concept*]

            # Set the embeddings at the concept dimension to the response values
            self.item_response_embeddings.weight.data[item_indices, :] = response_values_repeated

        # ------ None learnable parameters
        # Modality mask creation + mod_per_item



        self.register_buffer('nb_modalities',
                             torch.zeros(self.item_n, dtype=torch.long, device=self.device))  # without sentinels
        self.register_buffer('mask', torch.ones(self.item_n, self.nb_mod_max_plus_sent) * float('inf'))
        self.register_buffer('diff_mask', torch.zeros(self.item_n, self.nb_mod_max_plus_sent - 1))
        self.register_buffer('diff_mask2', torch.zeros(self.item_n, self.nb_mod_max_plus_sent - 2))

        if not load_params:

            R_t = self.R.clone()
            R_t[R_t == 0] = valid_data.log_tensor[R_t == 0]
            R_t = R_t.T - 1

            for item_i, logs in enumerate(R_t):
                unique_logs = torch.unique(logs)
                delta_min = torch.min(
                    torch.abs(unique_logs.unsqueeze(0) - unique_logs.unsqueeze(1)) + torch.eye(unique_logs.shape[0],
                                                                                               device=self.device))

                if delta_min < 1 / (self.nb_mod_max - 1):
                    self.nb_modalities[item_i] = self.nb_mod_max
                else:
                    self.nb_modalities[item_i] = (torch.round(1 / delta_min) + 1).long()

                self.mask[item_i, torch.arange(1, self.nb_modalities[item_i] + 1)] = 0
                self.diff_mask[item_i, torch.arange(self.nb_modalities[item_i] + 1)] = 1
                self.diff_mask2[item_i, torch.arange(self.nb_modalities[item_i])] = 1

            self.register_buffer('in_idx', torch.arange(self.item_n, device=self.device).unsqueeze(
                1) * self.nb_mod_max_plus_sent + self.nb_modalities.unsqueeze(1) + 1, persistent=False)
            self.register_buffer('ir_idx', resp_to_mod(self.R, self.nb_modalities), persistent=False)
        else :
            self.register_buffer('in_idx', None, persistent=False)
            self.register_buffer('ir_idx', None, persistent=False)

        # Indexes precomputing
        self.register_buffer('im_idx',
                             torch.arange(self.item_n, device=self.device).unsqueeze(
                                 1) * self.nb_mod_max_plus_sent + torch.arange(
                                 self.nb_mod_max_plus_sent, device=self.device).expand(self.item_n,
                                                                                         self.nb_mod_max_plus_sent), persistent=False)
        self.register_buffer('i0_idx',
                             torch.arange(self.item_n, device=self.device).unsqueeze(1) * self.nb_mod_max_plus_sent, persistent=False)


    def get_embeddings(self, user_ids, item_ids, concept_ids):
        # User embeddings
        u_emb = self.users_emb(user_ids)  # [batch_size, embedding_dim]

        # Compute item-response indices
        im_idx = self.im_idx[item_ids]  # [batch_size, nb_mod]
        i0_idx = self.i0_idx[item_ids]  # [batch_size, nb_mod]
        in_idx = self.in_idx[item_ids]  # [batch_size, nb_mod]

        # Item-Response embeddings
        im_emb_prime = self.item_response_embeddings(im_idx)  # [batch_size, nb_mod, embedding_dim]
        i0_emb_prime = self.item_response_embeddings(i0_idx)
        in_emb_prime = self.item_response_embeddings(in_idx)

        return u_emb, im_emb_prime, i0_emb_prime, in_emb_prime, None

    def forward(self, user_ids, item_ids, concept_ids):
        # I
        im_idx = self.im_idx[item_ids]
        im_emb = self.item_response_embeddings(im_idx)

        # E
        u_emb = self.users_emb(user_ids)

        # p_uim
        diff = u_emb.unsqueeze(1) - im_emb
        p_uim = torch.sum(diff ** 2, dim=2)

        return mod_to_resp(torch.argmin(p_uim + self.mask[item_ids, :], dim=1), self.nb_modalities[item_ids])

    def get_regularizer(self):
        return self.users_emb.weight.norm().pow(2) + self.item_response_embeddings.weight.norm().pow(
            2)

@torch.jit.export
class IMPACTModel_low_mem(nn.Module):
    '''
    Graph Convolutional Cognitive Diagnostic
    '''

    def __init__(self, user_n: int, item_n: int, concept_n: int, concept_map: dict, train_data: Dataset,
                 valid_data: Dataset,
                 d_in: int = 3, nb_mod_max: int = 12, load_params: bool = False):
        super(IMPACTModel_low_mem, self).__init__()
        self.user_n: int = user_n
        self.item_n: int = item_n
        self.nb_mod_max: int = nb_mod_max  # Discretized responses: 0.0 to 1.0 in steps of 0.1
        self.nb_mod_max_plus_sent: int = nb_mod_max + 2
        self.concept_n: int = concept_n
        self.concept_map: dict = concept_map

        # Register R as a buffer to ensure it's on the correct device
        self.register_buffer('R', train_data.log_tensor.clone())
        self.device = self.R.device

        # ------ Declare learnable parameters
        ## User embeddings
        self.users_emb = nn.Embedding(user_n, concept_n)
        ## Item-Response embeddings
        self.item_response_embeddings = nn.Embedding(item_n * self.nb_mod_max_plus_sent, d_in)
        ## Linear projection tensor
        self.W = nn.Parameter(torch.Tensor(concept_n, d_in, concept_n))

        # ------ Initialize Parameters
        ## Initialize W
        torch.nn.init.xavier_uniform_(self.W.data)
        F.normalize(self.W.data, p=2, dim=[1, 2], out=self.W.data)

        ## Initialize users and item-response embeddings
        self.users_emb.weight.data = self.users_emb.weight.data.zero_().to(self.device)
        self.item_response_embeddings.weight.data = self.item_response_embeddings.weight.data.zero_().to(self.device)

        k2q = defaultdict(set)
        for item, concept_list in concept_map.items():
            for concept in concept_list:
                k2q[concept].add(item)
        items_by_concepts = list(map(set, k2q.values()))
        for i, c in enumerate(items_by_concepts):
            c_list = torch.tensor(list(c), dtype=torch.long, device=self.device)
            self.users_emb.weight.data[:, i].add_(
                (self.R[:, c_list].sqrt().sum(dim=1) / (torch.sum(self.R[:, c_list] != 0, dim=1) + 1e-12))
            )

        response_values = torch.linspace(1, 2, steps=self.nb_mod_max_plus_sent, device=self.device)
        for concept_index, items in enumerate(k2q.values()):
            # Convert items to a tensor
            items = torch.tensor(list(items), dtype=torch.long, device=self.device)

            # Compute item-response indices for all responses
            # item_indices shape: [nb_items_in_concept, nb_responses]
            item_indices = items.unsqueeze(1) * self.nb_mod_max_plus_sent + torch.arange(self.nb_mod_max_plus_sent,
                                                                                         device=self.device).unsqueeze(
                0)

            # Flatten to a 1D tensor
            item_indices = item_indices.reshape(-1)  # [nb_items_in_concept * nb_mod_max]

            # Repeat response values for each item
            response_values_repeated = response_values.repeat(len(items)).unsqueeze(
                1)  # [nb_mod*nb_items_in_concept*]

            # Set the embeddings at the concept dimension to the response values
            self.item_response_embeddings.weight.data[item_indices, :] = response_values_repeated

        # ------ None learnable parameters
        # Modality mask creation + mod_per_item
        R_t = self.R.clone()
        R_t[R_t == 0] = valid_data.log_tensor[R_t == 0]
        R_t = R_t.T - 1
        self.register_buffer('nb_modalities',
                             torch.zeros(self.item_n, dtype=torch.long, device=self.device))  # without sentinels
        self.register_buffer('mask', torch.ones(self.item_n, self.nb_mod_max_plus_sent) * float('inf'))
        self.register_buffer('diff_mask', torch.zeros(self.item_n, self.nb_mod_max_plus_sent - 1))
        self.register_buffer('diff_mask2', torch.zeros(self.item_n, self.nb_mod_max_plus_sent - 2))

        for item_i, logs in enumerate(R_t):
            unique_logs = torch.unique(logs)
            delta_min = torch.min(
                torch.abs(unique_logs.unsqueeze(0) - unique_logs.unsqueeze(1)) + torch.eye(unique_logs.shape[0],
                                                                                           device=self.device))

            if delta_min < 1 / (self.nb_mod_max - 1):
                self.nb_modalities[item_i] = self.nb_mod_max
            else:
                self.nb_modalities[item_i] = (torch.round(1 / delta_min) + 1).long()

            self.mask[item_i, torch.arange(1, self.nb_modalities[item_i] + 1)] = 0
            self.diff_mask[item_i, torch.arange(self.nb_modalities[item_i] + 1)] = 1
            self.diff_mask2[item_i, torch.arange(self.nb_modalities[item_i])] = 1

        # Indexes precomputing
        self.register_buffer('im_idx',
                             torch.arange(self.item_n, device=self.device).unsqueeze(
                                 1) * self.nb_mod_max_plus_sent + torch.arange(
                                 self.nb_mod_max_plus_sent, device=self.device).expand(self.item_n,
                                                                                         self.nb_mod_max_plus_sent), persistent=False)
        self.register_buffer('i0_idx',
                             torch.arange(self.item_n, device=self.device).unsqueeze(1) * self.nb_mod_max_plus_sent, persistent=False)
        self.register_buffer('in_idx', torch.arange(self.item_n, device=self.device).unsqueeze(
            1) * self.nb_mod_max_plus_sent + self.nb_modalities.unsqueeze(1) + 1, persistent=False)

        self.register_buffer('ir_idx', resp_to_mod(self.R, self.nb_modalities), persistent=False)

    def get_embeddings(self, user_ids, item_ids, concept_ids):
        # User embeddings
        u_emb = self.users_emb(user_ids)  # [batch_size, embedding_dim]

        # Compute item-response indices
        im_idx = self.im_idx[item_ids]  # [batch_size, nb_mod]
        i0_idx = self.i0_idx[item_ids]  # [batch_size, nb_mod]
        in_idx = self.in_idx[item_ids]  # [batch_size, nb_mod]

        # Item-Response embeddings
        im_emb_prime = self.item_response_embeddings(im_idx)  # [batch_size, nb_mod, embedding_dim]
        i0_emb_prime = self.item_response_embeddings(i0_idx)
        in_emb_prime = self.item_response_embeddings(in_idx)

        # Compute negative squared Euclidean distances (p_uir)
        W_t = self.W[concept_ids]

        return u_emb, im_emb_prime, i0_emb_prime, in_emb_prime, W_t

    def get_regularizer(self):
        return self.users_emb.weight.norm().pow(2) + self.item_response_embeddings.weight.norm().pow(
            2) + self.W.data.norm().pow(2)

    def forward(self, user_ids, item_ids, concept_ids):
        # W
        W_t = self.W[concept_ids]

        # I
        im_idx = self.im_idx[item_ids]
        im_emb_prime = self.item_response_embeddings(im_idx)
        im_emb = torch.bmm(im_emb_prime, W_t)

        # E
        u_emb = self.users_emb(user_ids)

        # p_uim
        diff = u_emb.unsqueeze(1) - im_emb
        p_uim = torch.sum(diff ** 2, dim=2)

        return mod_to_resp(torch.argmin(p_uim + self.mask[item_ids, :], dim=1), self.nb_modalities[item_ids])

class IMPACT(AbstractModel):

    def __init__(self, **config):
        super().__init__('IMPACT', **config)
        self.L_W = torch.jit.script(CoVWeightingLoss(device=config['device']))

    def init_model(self, train_data: Dataset, valid_data: Dataset):
        self.concept_map = train_data.concept_map

        if self.config['low_mem'] == True:
            self.model = IMPACTModel_low_mem(train_data.n_users, train_data.n_items, train_data.n_categories, self.concept_map,
                                 train_data, valid_data, self.config['d_in'], self.config['num_responses'], load_params=self.config['load_params'])
            self.loss = custom_loss_low_mem
        else:
            self.model = IMPACTModel(train_data.n_users, train_data.n_items, train_data.n_categories, self.concept_map,
                                 train_data, valid_data, self.config['num_responses'], load_params=self.config['load_params'])
            self.loss = custom_loss

        super().init_model(train_data, valid_data)

    def _load_model_params(self, temporary=True) -> None:
        super()._load_model_params(temporary)

        self.model.in_idx = torch.arange(self.model.item_n, device=self.model.device).unsqueeze(
            1) * self.model.nb_mod_max_plus_sent + self.model.nb_modalities.unsqueeze(1) + 1
        self.model.ir_idx = resp_to_mod(self.model.R, self.model.nb_modalities)

    def _save_model_params(self, temporary=True) -> None:
        super()._save_model_params(temporary)


    def _loss_function(self, pred, real):
        return torch.tensor([4])

    @AbstractModel.evaluation_state
    def evaluate_valid(self, valid_dataloader: data.DataLoader, valid_tensor):

        loss_list = []
        pred_list = []
        label_list = []

        for data_batch in valid_dataloader:
            user_ids = data_batch[:, 0].long()
            item_ids = data_batch[:, 1].long()

            labels = data_batch[:, 2]
            concept_ids = data_batch[:, 3].long()

            preds = self.model(user_ids, item_ids, concept_ids)
            total_loss = self._compute_loss(user_ids, item_ids, concept_ids, labels)
            loss_list.append(total_loss.detach())

            pred_list.append(preds)
            label_list.append(labels)

        pred_tensor = torch.cat(pred_list)
        label_tensor = torch.cat(label_list)
        mean_loss = torch.mean(torch.stack(loss_list))

        return mean_loss, self.valid_metric(pred_tensor, label_tensor)

    def _compute_loss(self, users_id, items_id, concepts_id, labels):
        device = self.config['device']
        beta = 0.5

        lambda_param = self.config['lambda']

        u_emb, im_emb_prime, i0_emb_prime, in_emb_prime, W_t = self.model.get_embeddings(users_id, items_id,
                                                                                         concepts_id)

        L1, L2, L3 = self.loss(u_emb=u_emb, im_emb_prime=im_emb_prime, i0_emb_prime=i0_emb_prime,
                                 in_emb_prime=in_emb_prime, W_t=W_t,
                                 modalities_idx=self.model.ir_idx[users_id, items_id],
                                 nb_mod_max_plus_sent=self.model.nb_mod_max_plus_sent,
                                 diff_mask=self.model.diff_mask[items_id],
                                 diff_mask2=self.model.diff_mask2[items_id],
                                 users_id=users_id, items_id=items_id,
                                 concepts_id=concepts_id, R=self.model.R, users_emb=self.model.users_emb.weight)

        R = self.model.get_regularizer()

        # Stack losses into a tensor
        losses = torch.stack([L1, L3])  # Shape: (4,)

        # Update statistics and compute weights
        weights = self.L_W.compute_weights(losses)

        # Compute total loss
        total_loss = torch.dot(weights, losses) + beta * L2 + lambda_param * R

        return total_loss

    def get_user_emb(self):
        super().get_user_emb()
        return self.model.users_emb.weight.data


@torch.jit.script
def custom_loss_low_mem(u_emb: torch.Tensor,
                        im_emb_prime: torch.Tensor,
                        i0_emb_prime: torch.Tensor,
                        in_emb_prime: torch.Tensor,
                        W_t: torch.Tensor,
                        modalities_idx: torch.Tensor,
                        nb_mod_max_plus_sent: int,
                        diff_mask: torch.Tensor,
                        diff_mask2: torch.Tensor,
                        users_id: torch.Tensor,
                        items_id: torch.Tensor,
                        concepts_id: torch.Tensor,
                        R: torch.Tensor,
                        users_emb: torch.Tensor):
    im_emb = torch.bmm(im_emb_prime, W_t)
    i0_emb = torch.bmm(i0_emb_prime, W_t)
    in_emb = torch.bmm(in_emb_prime, W_t)

    diff = u_emb.unsqueeze(1) - im_emb
    diff_2 = i0_emb - im_emb
    diff_3 = in_emb - im_emb

    p_uir = -torch.sum(diff ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]
    p_uir_2 = -torch.sum(diff_2 ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]
    p_uir_3 = -torch.sum(diff_3 ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]

    ##### L1
    device = p_uir.device

    # Compute differences between adjacent modalities
    diffs = p_uir[:, :-1] - p_uir[:, 1:]  # shape = [batch_size, nb_mod_max_plus_sent-1]

    # Compute loss terms for responses greater and less than r
    greater_mask = torch.arange(nb_mod_max_plus_sent - 1, device=device).unsqueeze(0) >= modalities_idx.unsqueeze(
        1)  # nb_mod_max_plus_sent - 1 : start at 0 (torch.arange ok), sentinels (included)
    less_mask = ~greater_mask

    L1 = torch.where(diff_mask == 1, F.softplus((less_mask.int() - greater_mask.int()) * diffs),
                     torch.zeros_like(diff_mask)).mean(dim=1).mean()

    ##### L2
    diffs2 = (p_uir_2[:, 1:-1] - p_uir_2[:, 2:])
    diffs3 = (p_uir_3[:, :-2] - p_uir_3[:, 1:-1])

    L2 = torch.where(diff_mask2 == 1, F.softplus(diffs3 - diffs2), torch.zeros_like(diff_mask2)).mean(dim=1).mean()

    ##### L3
    R_t = R[users_id][:, items_id].t()
    b = (R[users_id, items_id].unsqueeze(1) - R_t)

    b_diag = b.abs()

    v_mask = concepts_id.unsqueeze(0).eq(concepts_id.unsqueeze(1))  # same concept checking
    u_mask = v_mask & (b_diag > 0.0) & (R_t >= 1.0)   # b_diag > 0 : not exactly similar responses for which we cannot say anything; R_t >= 1.0 : comparison with not null responses only
    
    indices = torch.nonzero(u_mask)
    u_base_idx = indices[:, 0]
    u_comp_idx = indices[:, 1]

    u_base_emb = users_emb[users_id[u_base_idx], concepts_id[u_base_idx]]
    u_comp_emb = users_emb[users_id[u_comp_idx], concepts_id[u_comp_idx]]

    sign_b = b[u_base_idx, u_comp_idx].sign()
    diff_emb = u_base_emb - u_comp_emb
    q_val = 1.0 - sign_b * diff_emb

    L3 = torch.clamp(q_val, min=0.0).mean()
    return L1, L2, L3

@torch.jit.script
def custom_loss(u_emb: torch.Tensor,
                        im_emb_prime: torch.Tensor,
                        i0_emb_prime: torch.Tensor,
                        in_emb_prime: torch.Tensor,
                        W_t: torch.Tensor,
                        modalities_idx: torch.Tensor,
                        nb_mod_max_plus_sent: int,
                        diff_mask: torch.Tensor,
                        diff_mask2: torch.Tensor,
                        users_id: torch.Tensor,
                        items_id: torch.Tensor,
                        concepts_id: torch.Tensor,
                        R: torch.Tensor,
                        users_emb: torch.Tensor):


    diff = u_emb.unsqueeze(1) - im_emb_prime
    diff_2 = i0_emb_prime - im_emb_prime
    diff_3 = in_emb_prime - im_emb_prime

    p_uir = -torch.sum(diff ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]
    p_uir_2 = -torch.sum(diff_2 ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]
    p_uir_3 = -torch.sum(diff_3 ** 2, dim=2)  # [batch_size, nb_mod_max_plus_sent]

    ##### L1
    device = p_uir.device

    # Compute differences between adjacent modalities
    diffs = p_uir[:, :-1] - p_uir[:, 1:]  # shape = [batch_size, nb_mod_max_plus_sent-1]

    # Compute loss terms for responses greater and less than r
    greater_mask = torch.arange(nb_mod_max_plus_sent - 1, device=device).unsqueeze(0) >= modalities_idx.unsqueeze(
        1)  # nb_mod_max_plus_sent - 1 : start at 0 (torch.arange ok), sentinels (included)
    less_mask = ~greater_mask

    L1 = torch.where(diff_mask == 1, F.softplus((less_mask.int() - greater_mask.int()) * diffs),
                     torch.zeros_like(diff_mask)).mean(dim=1).mean()

    ##### L2
    diffs2 = (p_uir_2[:, 1:-1] - p_uir_2[:, 2:])
    diffs3 = (p_uir_3[:, :-2] - p_uir_3[:, 1:-1])

    L2 = torch.where(diff_mask2 == 1, F.softplus(diffs3 - diffs2), torch.zeros_like(diff_mask2)).mean(dim=1).mean()

    ##### L3
    R_t = R[users_id][:, items_id].t()
    b = (R[users_id, items_id].unsqueeze(1) - R_t)

    b_diag = b.abs()

    v_mask = concepts_id.unsqueeze(0).eq(concepts_id.unsqueeze(1))  # same concept checking
    u_mask = v_mask & (b_diag > 0.0) & (R_t >= 1.0)   # b_diag > 0 : not exactly similar responses for which we cannot say anything; R_t >= 1.0 : comparison with not null responses only

    indices = torch.nonzero(u_mask)
    u_base_idx = indices[:, 0]
    u_comp_idx = indices[:, 1]

    u_base_emb = users_emb[users_id[u_base_idx], concepts_id[u_base_idx]]
    u_comp_emb = users_emb[users_id[u_comp_idx], concepts_id[u_comp_idx]]

    sign_b = b[u_base_idx, u_comp_idx].sign()
    diff_emb = u_base_emb - u_comp_emb
    q_val = 1.0 - sign_b * diff_emb

    L3 = torch.clamp(q_val, min=0.0).mean()
    return L1, L2, L3


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
