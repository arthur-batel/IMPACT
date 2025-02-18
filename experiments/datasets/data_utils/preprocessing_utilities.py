from collections import defaultdict
from typing import List

import pandas as pd
import random
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

def stat_unique(data: pd.DataFrame, key):
    """
    Calculate statistics on unique values in a DataFrame column or combination of columns.

    :param data: A pandas DataFrame.
    :type data: pd.DataFrame
    :param key: The column name or list of column names
    :type key: str or List[str]
    :return: None
    :rtype: None
    """
    if key is None:
        print('Total length: {}'.format(len(data)))
    elif isinstance(key, str):
        print('Number of unique {}: {}'.format(key, len(data[key].unique())))
    elif isinstance(key, list):
        print('Number of unique [{}]: {}'.format(','.join(key), len(data.drop_duplicates(key, keep='first'))))
def remove_duplicates(data: pd.DataFrame, key_attrs: List[str], agg_attrs: List[str]):
    """
    Remove duplicates from a DataFrame based on specified key attributes while aggregating other attributes.

    :param data: Dataset as a pandas.DataFrame
    :type data: pd.DataFrame
    :param key_attrs: Attributes whose combination must be unique
    :type key_attrs: List[str]
    :param agg_attrs: Attributes to aggregate in a set for every unique key
    :type agg_attrs: List[str]
    :return: DataFrame with duplicates removed and specified attributes aggregated
    :rtype: pd.DataFrame
    """
    special_attributes = key_attrs.copy()
    special_attributes.extend(agg_attrs)
    d = {}
    for agg_attr in agg_attrs:
        d.update({agg_attr: set})
    d.update({col: 'first' for col in data.columns if col not in special_attributes})
    data = data.groupby(key_attrs).agg(d).reset_index()
    return data

def densify(data: pd.DataFrame, grp_by_attr: str, count_attr: str, thd: int):
    """
    Filter out groups in a DataFrame based on a count threshold.

    :param data: Dataset
    :type data: pd.DataFrame
    :param grp_by_attr: Attribute used for grouping
    :type grp_by_attr: str
    :param count_attr: Attribute used for counting within groups
    :type count_attr: str
    :param thd: Threshold for group count (groups with less than thd filtered)
    :type thd: int
    :return: DataFrame with groups filtered out
    """
    n = data.groupby(grp_by_attr)[count_attr].nunique()
    filter = n[n < thd].index.tolist()
    print(f'filter {len(filter)} '+grp_by_attr)

    if len(filter) >0 :
        return data[~data[grp_by_attr].isin(filter)], len(filter)
    else : return data, 0

def create_q2k(data: pd.DataFrame):
    """
    Create mappings from item IDs to sets of dimension IDs and vice versa.

    :param data: Dataset containing 'item_id' and 'dimension_id' columns
    :type data: pd.DataFrame
    :return: item to knowledge mapping and knowledge to item mapping
    :rtype: Dict[str, Set[str]], Dict[str, Set[str]]
    """
    q2k = {}
    table = data.drop_duplicates(subset=["dimension_id","item_id"])
    for i, row in table.iterrows():
        q = int(row['item_id'])
        l = q2k.get(q,[])
        l.append(str(int(row['dimension_id'])))
        q2k[q] = l

    # get knowledge to item map
    k2q = {}
    for q, ks in q2k.items():
        for k in ks:
            k2q.setdefault(k, set())
            k2q[k].add(q)
    return q2k, k2q

def encode_attr(data: pd.DataFrame, attr:str):
    """
    Encode categorical attribute values with numerical IDs.

    :param data: Dataset
    :type data: pd.DataFrame
    :param attr: Attribute to renumber
    :type attr: str
    :return: Encoded DataFrame and mapping from attribute to numerical IDs
    :rtype: pd.DataFrame, Dict[str, int]
    """

    attr2n = {}
    cnt = 0
    for i, row in data.iterrows():
        if row[attr] not in attr2n:
            attr2n[row[attr]] = cnt
            cnt += 1

    data.loc[:, attr] = data.loc[:, attr].apply(lambda x: attr2n[x])
    return data.astype({attr:int}), attr2n


def parse_data(data):
    """
    Parse data into student-based and item-based datasets.

    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :return: Student-based and item-based datasets
    :rtype: defaultdict(dict), defaultdict(dict)
    """

    stu_data = defaultdict(lambda: defaultdict(dict))
    ques_data = defaultdict(lambda: defaultdict(dict))
    for i, row in data.iterrows():
        sid = row.user_id
        qid = row.item_id
        correct = row.correct
        stu_data[sid][qid] = correct
        ques_data[qid][sid] = correct
    return stu_data, ques_data

def quadruplet_format(data: pd.DataFrame):
    """
    Convert DataFrame into a list of quadruplets with correct data types.

    :param data: Dataset containing columns 'user_id', 'item_id', 'correct', and 'dimension_id'
    :type data: pd.DataFrame
    :return: List of quadruplets [sid, qid, score, dim]
    :rtype: list
    """
    # Ensure data types
    data['user_id'] = data['user_id'].astype(int)
    data['item_id'] = data['item_id'].astype(int)
    data['dimension_id'] = data['dimension_id'].astype(int)
    # 'correct' column might be float or int, depending on your data

    # Extract columns as lists
    user_ids = data['user_id'].tolist()
    item_ids = data['item_id'].tolist()
    corrects = data['correct'].tolist()
    dimension_ids = data['dimension_id'].tolist()

    # Combine columns into a list of quadruplets
    quadruplets = list(zip(user_ids, item_ids, corrects, dimension_ids))

    # Convert each quadruplet to a list
    quadruplets = [list(quad) for quad in quadruplets]

    return quadruplets




def one_hot_encoding(df,response_range_dict):
    # Pre-calculate num_copies for each q_name
    df = df[df['item_id'].isin(response_range_dict.keys())]

    df['r_range'] = df['item_id'].map(response_range_dict).astype(int)

    # Initialize an empty list to store the duplicated DataFrames
    dfs = []

    # Vectorized operation to duplicate rows
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        r_range = int(row['r_range'])
        c = int(row['correct'])

        # Create a binary list using numpy
        c_binary_list = np.zeros(r_range, dtype=int)
        c_binary_list[c - 1] = 1

        # Duplicate the row r_range times
        duplicated_rows = pd.DataFrame([row] * r_range)
        duplicated_rows['correct_binary'] = c_binary_list

        # Update item_id with the new values
        duplicated_rows['item_id'] = duplicated_rows['item_id'].astype(
            str) + '_' + duplicated_rows.index.astype(str)

        dfs.append(duplicated_rows)

    # Concatenate all the DataFrames in the list
    return pd.concat(dfs, ignore_index=True)

def rescaling_dict(metadata: pd.DataFrame,q2n):
    """
    Computes response_range_dict, min_response_dict and max_response_dict

    """
    response_range_dict = {}
    min_response_dict = {}
    max_response_dict = {}

    # Iterate over the DataFrame
    for i, row in metadata.iterrows():
        # Extract item ID, min_response, and max_response from the row
        try:
            item_id = q2n[row["Variable Name"]]
            min_response = row["min_response"]
            max_response = row["max_response"]

            # Calculate response range and store it in the dictionary
            response_range = max_response - min_response
            response_range_dict[item_id] = response_range
            min_response_dict[item_id] = min_response
            max_response_dict[item_id] = max_response
        except KeyError as e:
            print(f'{e} were removed from dataset')
    return [response_range_dict, min_response_dict, max_response_dict]

def split_data(triplets, stu_data, test_prop, valid_prop, least_test_length) :
    """
    Split data (list of triplets) into train, validation, and test sets.

    :param triplets: List of triplets (sid, qid, score)
    :type triplets: list
    :param stu_data: Student data
    :type stu_data:
    :param test_prop: Fraction of the test set
    :type test_prop: float
    :param valid_prop: Fraction of the validation set
    :type valid_prop: float
    :param least_test_length: Minimum number of items a student must have to be included in the test set.
    :type least_test_length: int or None
    :return: Train, validation, and test sets.
    :rtype: list, list, list
    """

    print("WARNING !!!! : least_test_length not used !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    n_students = len(stu_data)

    students = list(range(n_students))
    random.shuffle(students)
    if least_test_length is not None:
        student_lens = defaultdict(int)
        for t in triplets:
            student_lens[t[0]] += 1
        students = [student for student in students
                    if student_lens[student] >= least_test_length]
    n_students = len(students)    
    if isinstance(test_prop, float):
        test_prop = int(n_students * test_prop)

    if isinstance(valid_prop, float):
        valid_prop = int(n_students * valid_prop)

    train_size = n_students - test_prop
    assert (train_size > 0 and test_prop > 0)

    test_students = set(students[:test_prop])
    valid_students = set(students[test_prop:test_prop + valid_prop])
    train_students = set(students[test_prop + valid_prop:])

    train_data = [record for record in triplets if record[0] in train_students]
    valid_data = [record for record in triplets if record[0] in valid_students]
    test_data = [record for record in triplets if record[0] in test_students]

    return train_data, valid_data, test_data

def split_data_vertically(quadruplet, test_prop, valid_prop, folds_nb=5):
    """
    Split data (list of triplets) into train, validation, and test sets.

    :param quadruplet: List of triplets (sid, qid, score)
    :type quadruplet: list
    :param test_prop: Fraction of the test set
    :type test_prop: float
    :param valid_prop: Fraction of the validation set
    :type valid_prop: float
    :param least_test_length: Minimum number of items a student must have to be included in the test set.
    :type least_test_length: int or None
    :return: Train, validation, and test sets.
    :rtype: list, list, list
    """

    kf = KFold(n_splits=folds_nb, shuffle=True)
    df = pd.DataFrame(quadruplet)
    df.columns = ["student_id","item_id","answer","dimension_id"]

    train = [[]  for _ in range(folds_nb)]
    valid = [[]  for _ in range(folds_nb)]
    test = [[]  for _ in range(folds_nb)]

    for i_group, group in df.groupby('student_id'):
        group_idxs = np.array(group.index)

        for i_fold,(train_valid_fold_idx, test_fold_idx)  in enumerate(kf.split(group_idxs)):

            train_valid_item_idx = group_idxs[train_valid_fold_idx]
            test_item_idx = group_idxs[test_fold_idx]

            train_item_idx, valid_item_idx = train_test_split(train_valid_item_idx, test_size=float(valid_prop) / (
                        1.0 - float(test_prop)), shuffle=True)

            train[i_fold] += df.loc[train_item_idx, :].values.tolist()
            valid[i_fold] += df.loc[valid_item_idx, :].values.tolist()
            test[i_fold] += df.loc[test_item_idx, :].values.tolist()

    return train, valid, test

def split_small_data_vertically(quadruplet, test_prop, valid_prop):
    """
    Split data (list of quadruplets) into train, validation, and test sets.

    :param quadruplet: List of quadruplets (student_id, item_id, answer, dimension_id)
    :type quadruplet: list
    :param test_prop: Fraction of the test set
    :type test_prop: float
    :param valid_prop: Fraction of the validation set
    :type valid_prop: float
    :return: Train, validation, and test sets.
    :rtype: list, list, list
    """

    # Create DataFrame
    df = pd.DataFrame(quadruplet, columns=["student_id", "item_id", "answer", "dimension_id"])

    # Initialize lists to collect indices
    train_indices = []
    valid_indices = []
    test_indices = []

    # Group by 'student_id' and split data
    for _, group in df.groupby('student_id'):
        group_indices = group.index.values

        # Split into train+valid and test
        train_valid_idx, test_idx = train_test_split(
            group_indices,
            test_size=test_prop,
            shuffle=True
        )

        # Further split train_valid into train and valid
        valid_size = valid_prop / (1.0 - test_prop)
        train_idx, valid_idx = train_test_split(
            train_valid_idx,
            test_size=valid_size,
            shuffle=True
        )

        # Collect indices
        train_indices.extend(train_idx)
        valid_indices.extend(valid_idx)
        test_indices.extend(test_idx)

    # Extract data using the collected indices
    train_df = df.loc[train_indices]
    valid_df = df.loc[valid_indices]
    test_df = df.loc[test_indices]

    # Optionally reset index if needed
    # train_df = train_df.reset_index(drop=True)
    # valid_df = valid_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

    # Convert DataFrames to lists (if required)
    train = train_df.values.tolist()
    valid = valid_df.values.tolist()
    test = test_df.values.tolist()

    return train, valid, test


def save_to_csv(data, path):
    """
    Save list of triplets (sid, qid, score) to a CSV file.
    :param data: List of triplets (sid, qid, score)
    :type data: pd.DataFrame
    :param path: Path to CSV file
    :type path: str
    """
    pd.DataFrame.from_records(sorted(data), columns=['student_id', 'item_id', 'correct', 'dimension_id']).to_csv(path, index=False)

def get_metadata(data: pd.DataFrame, keys: List[str]) -> dict:
    m = {}
    for attr in keys:
        m["num_"+attr] = len(data[attr].unique())
    return m