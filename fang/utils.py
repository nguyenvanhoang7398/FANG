from sklearn.model_selection import train_test_split
from dataset import utils
import torch
import numpy as np
import scipy.sparse as sp
import re
from dataset.utils import *


NAME_WEIGHT_DELIMITER = "#"
STANCE_DELIMITER = "_"
USER_TAG, NEWS_TAG, SOURCE_TAG, COMMUNITY_TAG = "user", "news", "source", "community"


def tag(entity_type, entity):
    return "{}_{}".format(entity_type, entity)


def is_tag(entity_type, entity):
    return entity.startswith(entity_type)


def remove_tag(tagged_entity):
    return tagged_entity.split("_")[-1]


def remove_url(url_str):
    text = re.sub(r"http\S+", "", url_str)
    return text


def get_default_paths(config, is_community):
    # Declare path to different relation adjacency matrices
    ent_p = os.path.join(config.root, "entities.txt" if not is_community else "community_entities.txt")
    feat_p = os.path.join(config.root, "entity_features.tsv" if not is_community else "community_features.tsv")
    cite_p = os.path.join(config.root, "source_citation.tsv")
    publ_p = os.path.join(config.root, "source_publication.tsv")
    rela_p = os.path.join(config.root, "user_relationships.tsv" if not is_community else "community_relationships.tsv")
    retw_p = os.path.join(config.root, "user_retweets.tsv" if not is_community else "community_retweets.tsv")
    news_info_p = os.path.join(config.root, "news_info.tsv")

    rep_p = os.path.join(config.root, "report.tsv" if not is_community else "community_report.tsv")
    sup_p = os.path.join(config.root, "support.tsv" if not is_community else "community_support.tsv")
    deny_p = os.path.join(config.root, "deny.tsv" if not is_community else "community_deny.tsv")
    unrelat_p = os.path.join(config.root, "unrelated.tsv" if not is_community else "community_unrelated.tsv")
    neu_p = os.path.join(config.root, "comment_neutral.tsv" if not is_community else "community_neutral.tsv")
    neg_p = os.path.join(config.root, "comment_negative.tsv" if not is_community else "community_negative.tsv")

    neu_sup_p = os.path.join(config.root, "support_neutral.tsv")
    neg_sup_p = os.path.join(config.root, "support_negative.tsv")

    return ent_p, feat_p, cite_p, publ_p, \
        rela_p, retw_p, news_info_p, rep_p, \
        sup_p, deny_p, unrelat_p, neu_p, \
        neg_p, neu_sup_p, neg_sup_p


def get_news_label_map(news_info_path):
    news_info_data = utils.read_csv(news_info_path, True, "\t")
    news_label_map = {row[0]: row[1] for row in news_info_data}
    return news_label_map


def get_train_val_test_labels_nodes(config, entities, news_label_map, inference=False):
    if config.train_test_val is not None and not inference:
        idx_train, idx_val, idx_test = [], [], []
        train_test_path = os.path.join(config.root, config.train_test_val)
        train_test_val = load_json(train_test_path)
        train_news, val_news, test_news = train_test_val["train"], train_test_val["val"], train_test_val["test"]
        for i, e in enumerate(entities):
            if e in news_label_map:
                if e in train_news:
                    idx_train.append(i)
                elif e in val_news:
                    idx_val.append(i)
                elif e in test_news:
                    idx_test.append(i)
    # indicate inference mode, use all data for test
    elif inference:
        idx_train, idx_val, idx_test = [], [], range(len(news_label_map))
    else:
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        idx_train, idx_val_test = train_test_split(range(len(news_label_map)), train_size=train_ratio)
        idx_val, idx_test = train_test_split(idx_val_test, train_size=val_ratio / (val_ratio + test_ratio))
    return idx_train, idx_val, idx_test


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def row_normalize(mtx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mtx.sum(axis=1))
    sum_inv = np.power(row_sum, -1).flatten()
    sum_inv[np.isinf(sum_inv)] = 0
    row_mtx_inv = sp.diags(sum_inv)
    return row_mtx_inv.dot(mtx)


def encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def encode_class_idx_label(label_map):
    classes = list(sorted(set(label_map.values())))
    class2idx = {c: i for i, c in enumerate(classes)}
    labels_onehot_map = {k: class2idx[v] for k, v in label_map.items()}
    return labels_onehot_map, len(classes), class2idx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_meta(fang_config):
    meta_data_path = os.path.join(fang_config.root, fang_config.meta_data_path)
    if os.path.exists(meta_data_path):
        meta_data = read_csv(meta_data_path, True, "\t")
    else:
        meta_data = None
    return meta_data
