import logging
from collections import defaultdict, deque

import torch
from torch.utils import data
import numpy as np
import pandas as pd
from experiments.datasets.data_utils import preprocessing_utilities as pu

class Dataset(object):

    def __init__(self, data, concept_map, metadata):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        self.metadata = metadata

        self._raw_data = data
        self._concept_map = concept_map
        self._n_logs = self._raw_data.shape[0]
        # After self._raw_data = data
        self._raw_data_array = self._generate_raw_data_array()  # precompute right away
        self._log_tensor = self._generate_log_tensor()  # precompute right away
        self._sparse_log_tensor = None
        self._norm_log_tensor = None

        self._users_id = set(int(x[0]) for x in data) # Ids of the users in this dataset instance (after splitting)
        self._items_id = set(int(x[1]) for x in data) # Ids of the items in this dataset instance (after splitting)
        concepts_id = set(sum(concept_map.values(), []))
        self._concepts_id = {int(x) for x in concepts_id}



        assert max(self._users_id) < self.n_users, \
            f'Require item ids renumbered : max user id = {max(self._users_id)}; nb users = {self.n_users}'
        assert max(self._items_id) < self.n_items, \
            f'Require item ids renumbered : max item id = {max(self._items_id)}; nb items = {self.n_items}'
        assert max(self._concepts_id) < self.n_categories, \
            f'Require concept ids renumbered : max concept id = {max(concepts_id)}; nb concepts = {self.n_categories}'

    @property
    def n_users(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self.metadata["num_user_id"]

    @property
    def n_logs(self):
        """
        @return: Total number of users in the dataset (before splitting)
        """
        return self._n_logs

    @property
    def n_items(self):
        """
        @return: Total number of items in the dataset (before splitting)
        """
        return self.metadata["num_item_id"]

    @property
    def n_categories(self):
        """
        @return: Total number of categories
        """
        return self.metadata["num_dimension_id"]

    @property
    def users_id(self):
        """
        @return: Ids of the users in this dataset instance (after splitting)
        """
        return self._users_id

    @property
    def items_id(self):
        """
        @return: Ids of the items in this dataset instance (after splitting)
        """
        return self._items_id

    @property
    def concepts_id(self):
        """
        @return: Ids of the items in this dataset instance (after splitting)
        """
        return self._concepts_id

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def raw_data_array(self):
        """
        @return: An array containing the raw data (shape = (self.n_logs,3)).
        """
        if self._raw_data_array is None:
            self._raw_data_array = self._generate_raw_data_array()
        return self._raw_data_array

    @property
    def log_tensor(self):
        return self._log_tensor

    @property
    def sparse_log_tensor(self):
        return self._sparse_log_tensor

    @property
    def norm_log_tensor(self):
        return self._norm_log_tensor

    def normalize_log_tensor(self)->None:
        self._norm_log_tensor = torch.nn.functional.normalize(self.log_tensor, dim=(0,1))

    def sparsify_log_tensor(self)->None:
        self._sparse_log_tensor = self._log_tensor.to_sparse()

    def _generate_raw_data_array(self):
        tensor_data = torch.zeros(size=(self.n_logs, 4))
        for i, (row, col, val, d) in enumerate(self.raw_data):
            tensor_data[i, 0] = int(row)
            tensor_data[i, 1] = int(col)
            tensor_data[i, 2] = val
            tensor_data[i, 3] = int(d)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor_data = tensor_data.to(device)

        return tensor_data

    def _generate_log_tensor(self):
        tensor_data = torch.zeros((self.n_users, self.n_items), device=self.raw_data_array.device)
        sid = self.raw_data_array[:, 0].long()
        qid = self.raw_data_array[:, 1].long()
        val = self.raw_data_array[:, 2]

        tensor_data.index_put_((sid, qid), val)
        return tensor_data




class LoaderDataset(Dataset, data.dataset.Dataset):

    def __init__(self, data, concept_map, metadata):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts
        """
        super().__init__(data, concept_map, metadata)

    def __getitem__(self, item):
        return self.raw_data_array[item]

    def __len__(self):
        return len(self.raw_data)

class AdaptiveDataset(Dataset):

    def __init__(self, data, concept_map, metadata,meta_size = 0):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            metadata : dict of keys {"num_user_id", "num_item_id", "num_dimension_id"}, containing the total number of users, items and concepts

        """
        super().__init__(data, concept_map,metadata)

        # Data dictionary self._data[users][items] = response
        self._data = {}
        for sid, qid, response, _ in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = response

        # initialize selection_set, meta_set,  submitted_set and unsubmitted_set set
        self._submitted_set = None
        self._unsubmitted_set = None
        self._selection_set = None #self._selection_set = self._submitted_set union self._unsubmitted_set
        self._meta = None

        self._init_meta_set(meta_size)
        self.reset_submitted()

    @property
    def submitted_set(self):
        return self._submitted_set

    @property
    def unsubmitted_set(self):
        return self._unsubmitted_set

    @property
    def submitted_loader(self):
        return self._submitted_set

    @property
    def unsubmitted_loader(self):
        return self._unsubmitted_set

    def _init_meta_set(self,meta_size):
        self._meta = {}
        self._selection_set = {}

        for uid in self._data:
            user_items = np.array(list(self._data[uid]))

            if user_items.shape[0] < meta_size:
                logging.warning(f"User {uid} has {user_items.shape[0]} for a meta size of {meta_size}")

                self._selection_set[uid] = {}
                self._meta[uid] = {}
                for qid in user_items:
                    self._meta[uid][qid] = self._data[uid][qid]

            else :
                meta_user_items = np.random.choice(user_items, meta_size, replace=False)

                self._selection_set[uid] = {}
                self._meta[uid] = {}
                for qid in meta_user_items:
                    self._meta[uid][qid] = self._data[uid][qid]
                for qid in set(user_items).difference(set(meta_user_items)):
                    self._selection_set[uid][qid] = self._data[uid][qid]

    def reset_submitted(self):
        """
        Set submitted_set set empty
        """
        self._submitted_set = defaultdict(deque)
        self._unsubmitted_set = defaultdict(set)

        for uid in self._selection_set:
            self._unsubmitted_set[uid] = set(self._selection_set[uid].keys())

    def apply_selection(self, student_idx: int, items_idx: list or np.array):
        """
        Add unsubmitted_set items to the submitted_set set
        Args:
            student_idx: int
            items_idx: list or np.array
        """

        for q in items_idx:
            assert q in self._unsubmitted_set[student_idx], \
                'Selected item not allowed'
            self._unsubmitted_set[student_idx].remove(q)
            self._submitted_set[student_idx].append(q)

    def get_tested_dataset(self, last=False):
        """
        Get submitted_set data for training
        Args:
            last: bool, True - the last item, False - all the submitted_set items
        Returns:
            LoaderDataset
        """
        triplets = []
        for sid, qids in self._submitted_set.items():
            if last:
                qid = qids[-1]
                triplets.append((sid, qid, self._data[sid][qid]))
            else:
                for qid in qids:
                    triplets.append((sid, qid, self._data[sid][qid]))
        return LoaderDataset(triplets, self.concept_map, self.metadata)

    def get_untested_dataset(self, last=False):
        """
        Get submitted_set data for training
        Args:
            last: bool, True - the last item, False - all the submitted_set items
        Returns:
            LoaderDataset
        """
        triplets = []
        for sid, qids in self._unsubmitted_set.items():
            if last:
                qid = qids[-1]
                triplets.append((sid, qid, self._data[sid][qid]))
            else:
                for qid in qids:
                    triplets.append((sid, qid, self._data[sid][qid]))
        return LoaderDataset(triplets, self.concept_map, self.metadata)
