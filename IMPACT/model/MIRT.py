import functools
import os
from collections import defaultdict

from torch.masked import MaskedTensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch.utils.data as data

from IMPACT.model.abstract_model import AbstractModel
from IMPACT.dataset import *
import torch.nn.functional as F

import warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of MaskedTensors is in prototype stage and will change in the near future. Please open a Github issue for features requests and see our documentation on the torch.masked module for further information about the project.*",
    category=UserWarning
)


@torch.jit.export
class MIRT2PL(nn.Module):
    '''
    MIRT 2PL model + response aware initialization
    '''

    def __init__(self, user_n: int, item_n: int, concept_n: int, concept_map: dict, train_data: Dataset,):
        super(MIRT2PL, self).__init__()
        self.user_n: int = user_n
        self.item_n: int = item_n
        self.concept_n: int = concept_n
        self.concept_map: dict = concept_map

        # Register R as a buffer to ensure it's on the correct device
        self.register_buffer('R', train_data.log_tensor.clone())

        self.device = self.R.device

        # ------ Declare learnable parameters
        ## User embeddings
        self.users_emb = nn.Embedding(user_n, concept_n)
        ## Item-Response embeddings
        self.item_discrimination = nn.Embedding(item_n, concept_n)
        self.item_difficulty = nn.Embedding(item_n,1)

        # ------ Initialize Parameters

        ## Initialize users and item-response embeddings
        self.users_emb.weight.data = self.users_emb.weight.data.zero_().to(self.device)
        self.item_discrimination.weight.data = self.item_discrimination.weight.data.normal_().to(self.device)
        self.item_difficulty.weight.data = self.item_difficulty.weight.data.normal_().to(self.device)

        k2q = defaultdict(set)
        for item, concept_list in concept_map.items():
            for concept in concept_list:
                k2q[concept].add(item)
        items_by_concepts = list(map(set, k2q.values()))
        for i, q in enumerate(items_by_concepts):
            q_list = torch.tensor(list(q), dtype=torch.long, device=self.device)
            self.users_emb.weight.data[:, i].add_(
                (self.R[:, q_list].sqrt().sum(dim=1) / (torch.sum(self.R[:, q_list] != 0, dim=1) + 1e-12))
            )


    def forward(self, user_ids, item_ids, concept_ids):

        a = self.item_discrimination(item_ids)
        b = self.item_difficulty(item_ids)
        theta = self.users_emb(user_ids)

        return F.sigmoid(torch.dot(a, theta) - b)

    def get_regularizer(self):
        return self.users_emb.weight.norm().pow(2) + self.item_discrimination.weight.norm().pow(
            2) + self.item_difficulty.weight.norm().pow(2)

class MIRT(AbstractModel):

    def __init__(self, **config):
        super().__init__('MIRT', **config)

    def init_model(self, train_data: Dataset, valid_data: Dataset):
        self.concept_map = train_data.concept_map

        self.model = MIRT2PL(train_data.n_users, train_data.n_items, train_data.n_categories, self.concept_map,
                             train_data)

        self.loss= CrossEntropyLoss

        super().init_model(train_data, valid_data)

    def _load_model_params(self, temporary=True) -> None:
        super()._load_model_params(temporary)
        self.model.to(self.config['device'])

    def _save_model_params(self, temporary=True) -> None:
        super()._save_model_params(temporary)


    def _loss_function(self, pred, real):

        R = self.model.get_regularizer()
        return self.loss(pred, real) + self.config['lambda_'] * R

    def get_user_emb(self):
        super().get_user_emb()
        return self.model.users_emb.weight.data