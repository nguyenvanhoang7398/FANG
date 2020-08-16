import numpy as np
from scipy import special
import random
import torch
import torch.nn.functional as F
from torch import nn
import time
from dataset.utils import *
from collections import OrderedDict


# from dataset.utils import read_csv
# meta_data_path = "data/fang/meta_data.tsv"
# meta_data = read_csv(meta_data_path, False, "\t")
# community_map = {int(row[0]): row[-1][-4:] for row in meta_data}
LOGGING, LOG_PROB = False, 1.
CACHING = True

class LRUCache:

    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

            # first, we add / update the key by conventional methods.

    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value: set()) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

        # RUNNER


class GraphSageWalker(object):
    def __init__(self, adj_lists, stance_lists, news, users, sources, train_dev_test_nodes, news_labels,
                 device, cache_path, entities, max_engages=100):
        super(GraphSageWalker, self).__init__()
        self.Q = 10     # (10/20*)
        self.USER_WALK_LEN = 2
        # self.NEG_USER_WALK_LEN = 2
        self.SOURCE_NEWS_WALK_LEN = 1
        # self.NEG_SOURCE_NEWS_WALK_LEN = 2   # (2/3*)
        self.USER_NUM_NEG = 8    # (100/200*)
        self.USER_NUM_POS = 8
        self.SOURCE_NEWS_NUM_NEG = 4   # (50*/100)
        self.SOURCE_NEWS_NUM_POS = 4
        self.MARGIN = 3
        self.max_engages = max_engages
        self.adj_lists = adj_lists
        self.stance_lists = stance_lists
        self.news, self.users, self.sources = news, users, sources
        self.train_nodes, self.val_nodes, self.test_nodes = train_dev_test_nodes
        # include all nodes in sampling, have to be very careful about this
        # try switching to just train nodes later
        self.sampled_nodes = self.train_nodes | self.val_nodes | self.test_nodes
        self.news_labels = news_labels
        # self.sim_mtx = np.matmul(raw_features, raw_features.transpose())
        self.device = device

        self.positive_pairs = []
        self.negative_pairs = []
        self.node_positive_pairs = {}
        self.node_negative_pairs = {}
        self.unique_nodes_batch = []

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.source_news_cache_path = cache_path.format("source_news_" + str(self.SOURCE_NEWS_WALK_LEN))
        self.user_cache_path = cache_path.format("user_" + str(self.USER_WALK_LEN))
        self.source_news_neighbor_cache, self.user_neighbor_cache, self.write_cache = self.load_cache_far_nodes()
        self.entities = entities

    def save(self):
        if self.write_cache and CACHING:
            print("Saving cache")
            save_to_pickle((self.source_news_neighbor_cache.cache, self.source_news_neighbor_cache.capacity),
                           self.source_news_cache_path)
            save_to_pickle((self.user_neighbor_cache.cache, self.user_neighbor_cache.capacity), self.user_cache_path)
            self.write_cache = False

    def load_cache_far_nodes(self):

        if CACHING and os.path.exists(self.source_news_cache_path) and os.path.exists(self.user_cache_path):
            print("Loading cache")
            source_news_neighbor_cache_dict, source_news_cache_capacity = load_from_pickle(self.source_news_cache_path)
            user_neighbor_cache_dict, user_cache_capacity = load_from_pickle(self.user_cache_path)
            source_news_neighbor_cache = LRUCache(source_news_cache_capacity)
            source_news_neighbor_cache.cache = source_news_neighbor_cache_dict
            user_neighbor_cache = LRUCache(user_cache_capacity)
            user_neighbor_cache.cache = user_neighbor_cache_dict
            write_cache = False
        else:
            capacity = 20000
            print("Creating new far node cache with capacity {}".format(capacity))
            source_news_neighbor_cache, user_neighbor_cache = LRUCache(capacity), LRUCache(capacity)
            write_cache = True
        return source_news_neighbor_cache, user_neighbor_cache, write_cache

    def compute_stance_loss(self, logits, stance_labels_tensor):
        total_ce_loss = self.cross_entropy(logits, stance_labels_tensor)
        return total_ce_loss.mean()

    def compute_news_loss(self, logits, news_labels_tensor):
        total_ce_loss = self.cross_entropy(logits, news_labels_tensor)
        return total_ce_loss.mean()

    def compute_unsup_loss_normal(self, embeddings, nodes):
        # assert len(embeddings) == len(self.unique_nodes_batch)
        # assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        # node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}
        node2index = {n: i for i, n in enumerate(nodes)}

        node_scores = []
        # assert len(self.node_positive_pairs) == len(self.node_negative_pairs)
        # assert len(self.node_positive_pairs) > 0
        # print("Number of nodes: positive pairs {}, negative pairs {}".format(len(self.node_positive_pairs),
        #                                                                      len(self.node_negative_pairs)))
        # for node in self.node_positive_pairs:
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negative_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            neg_indexs = [list(x) for x in zip(*nps)]
            neg_node_indexs = [node2index[x] for x in neg_indexs[0]]
            neg_neighb_indexs = [node2index[x] for x in neg_indexs[1]]
            neg_node_embeddings, neg_neighb_embeddings = embeddings[neg_node_indexs], embeddings[neg_neighb_indexs]
            neg_sim_score = F.cosine_similarity(neg_node_embeddings, neg_neighb_embeddings)
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_sim_score)), 0)

            # multiple positive score
            pos_indexs = [list(x) for x in zip(*pps)]
            pos_node_indexs = [node2index[x] for x in pos_indexs[0]]
            pos_neighb_indexs = [node2index[x] for x in pos_indexs[1]]
            pos_node_embeddings, pos_neighb_embeddings = embeddings[pos_node_indexs], embeddings[pos_neighb_indexs]
            pos_sim_score = F.cosine_similarity(pos_node_embeddings, pos_neighb_embeddings)
            pos_score = torch.log(torch.sigmoid(pos_sim_score))

            # proximity loss
            node_score = torch.mean(- pos_score - neg_score).view(1, -1)
            node_scores.append(node_score)

        if len(node_scores) > 0:
            loss = torch.mean(torch.cat(node_scores, 0))
        else:
            loss = None
        return loss

    def compute_unsup_loss_margin(self, embeddings, nodes):
        # assert len(embeddings) == len(self.unique_nodes_batch)
        # assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        # node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}
        node2index = {n: i for i, n in enumerate(nodes)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negative_pairs)
        # for node in self.node_positive_pairs:
        for node in nodes:
            pps = self.node_positive_pairs[node]
            nps = self.node_negative_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(
                torch.max(torch.tensor(0.0).to(self.device), neg_score - pos_score + self.MARGIN).view(1, -1))
        # nodes_score.append((-pos_score - neg_score).view(1,-1))

        if len(nodes_score) > 0:
            loss = torch.mean(torch.cat(nodes_score, 0), 0)
        else:
            loss = None
        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

        return loss

    def fetch_news_user_stance(self, nodes):
        news, users, stance_labels = [], [], []
        for node in nodes:
            stance_neighs = self.stance_lists[node]
            if node in self.news:
                for neigh, (stance, _, _) in stance_neighs:
                    news.append(node)
                    users.append(neigh)
                    stance_labels.append(stance)
            elif node in self.users:
                for neigh, (stance, _, _) in stance_neighs:
                    news.append(neigh)
                    users.append(node)
                    stance_labels.append(stance)
        return news, users, stance_labels

    def fetch_news_user_stance2(self, news_nodes):
        """
        We fetch data required for stance prediction
        """
        all_engaged_users, all_stance_labels, all_n_users = [], [], []
        for news in news_nodes:
            stance_info = self.stance_lists[news]
            for neigh, (stance, _, _) in stance_info:
                stance_idx = np.argmax(stance)
                all_engaged_users.append(neigh)
                all_stance_labels.append(stance_idx)
            all_n_users.append(len(stance_info))
        return all_engaged_users, all_stance_labels, all_n_users

    def export_eng_dist(self, nodes, bin_size, n_bins):
        max_duration = (bin_size * n_bins) - 1
        eng_dist_arr = []
        for node in nodes:
            if node in self.news_labels:
                bins = np.zeros(n_bins)
                for user, (stance, ts_mean, ts_std) in self.stance_lists[node]:
                    bin_idx = int(min(ts_mean, max_duration-1) / bin_size)
                    bins[bin_idx] += 1
                if np.sum(bins) > 0:
                    bins /= np.sum(bins)
                    eng_dist_arr.append(bins)
        eng_dist = np.sum(eng_dist_arr, axis=0)
        eng_dist /= np.sum(eng_dist)
        return eng_dist

    def fetch_news_classification(self, nodes, n_stances, cutoff_range=None, max_duration=86400*7):
        smoothing_coeff = 1e-8  # to differentiate between 0 ts and padding value
        sources, news = [], []
        all_engage_users, all_engage_stances, all_engage_ts, all_masked_attn, labels = [], [], [], [], []
        n_skips = []
        for node in nodes:
            if node in self.news_labels:
                skips = 0
                news.append(node)
                _sources = self.get_neigh_weights(node, node_only=True)
                assert len(_sources) == 1, "Only 1 source can publish the article"
                sources.append(_sources[0])
                engage_users, engage_stances, engage_ts, masked_attn = [], [], [], []
                for user, (stance, ts_mean, ts_std) in self.stance_lists[node]:
                    ts_mean = min(ts_mean, max_duration)
                    scaled_ts_mean = min(ts_mean, max_duration) / float(max_duration)
                    assert scaled_ts_mean >= 0
                    if cutoff_range is not None and (cutoff_range[0] <= ts_mean <= cutoff_range[1]):
                        skips += 1
                        continue
                    engage_users.append(user)
                    engage_stances.append(stance)
                    engage_ts.append([scaled_ts_mean + smoothing_coeff, ts_std + smoothing_coeff])
                    masked_attn.append(0)
                engage_users, engage_stances, engage_ts, masked_attn = \
                    engage_users[:self.max_engages], engage_stances[:self.max_engages], engage_ts[:self.max_engages], \
                    masked_attn[:self.max_engages]
                while len(engage_stances) < self.max_engages:
                    engage_stances.append(list(np.zeros(n_stances)))
                    engage_ts.append([0, 0])
                    masked_attn.append(-1000)
                all_engage_users.append(engage_users)
                all_engage_stances.append(engage_stances)
                all_engage_ts.append(engage_ts)
                all_masked_attn.append(masked_attn)
                labels.append(self.news_labels[node])
                n_skips.append(skips)
        return sources, news, all_engage_users, all_engage_stances, all_engage_ts, all_masked_attn, labels

    def extend_nodes(self, nodes):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negative_pairs = []
        self.node_negative_pairs = {}

        # Old code to sample and aggregate with positive and negative separately
        # # s_t = time.time()
        source_news_nodes = [node for node in nodes if (node in self.sources or node in self.news)]
        # self.get_positive_nodes(source_news_nodes, self.N_SOURCE_NEWS_WALK, self.SOURCE_NEWS_WALK_LEN)
        user_nodes = [node for node in nodes if node in self.users]
        # self.get_positive_nodes(user_nodes, self.N_USER_WALKS, self.USER_WALK_LEN)
        # # e_t = time.time()
        # # print("Positive sampling time: {:.4f}s".format(e_t - s_t))
        # # print("Sampling negative nodes for sources and news")
        # self.get_negative_nodes(source_news_nodes, self.SOURCE_NEWS_NUM_NEG, self.NEG_WALK_LEN)
        # # print("Sampling negative nodes for users")
        # self.get_negative_nodes(user_nodes, self.USER_NUM_NEG, self.NEG_WALK_LEN)
        # # print("Negative sampling time: {:.4f}s".format(time.time() - e_t))
        self.get_proximity_samples(source_news_nodes, self.SOURCE_NEWS_NUM_POS, self.SOURCE_NEWS_NUM_NEG,
                                   self.SOURCE_NEWS_WALK_LEN, self.source_news_neighbor_cache)
        self.get_proximity_samples(user_nodes, self.USER_NUM_POS, self.USER_NUM_NEG,
                                   self.USER_WALK_LEN, self.user_neighbor_cache)
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negative_pairs for i in x]))
        return self.unique_nodes_batch

    def extend_source_news_nodes(self, source_news_nodes):
        self.get_proximity_samples(source_news_nodes, self.SOURCE_NEWS_NUM_POS, self.SOURCE_NEWS_NUM_NEG,
                                   self.SOURCE_NEWS_WALK_LEN, self.source_news_neighbor_cache)

    def extend_user_nodes(self, user_nodes):
        self.get_proximity_samples(user_nodes, self.USER_NUM_POS, self.USER_NUM_NEG,
                                   self.USER_WALK_LEN, self.user_neighbor_cache)

    def reset(self):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negative_pairs = []
        self.node_negative_pairs = {}

    def get_unique_node_batch(self):
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negative_pairs for i in x]))
        return self.unique_nodes_batch

    def get_proximity_samples(self, nodes, num_pos, num_neg, neg_walk_len, neighbor_cache):
        # for optimization, we perform both positive and negative sampling in a single walk
        source_news = self.news | self.sources

        for node in nodes:
            homo_nodes = self.users if node in self.users else source_news
            neighbors = neighbor_cache.get(node)
            if neighbors == -1:
                neighbors, frontier = {node}, {node}
                for _ in range(neg_walk_len):
                    current = set()
                    for outer in frontier:
                        # we get all the neighbors of each frontier node, this may include nodes that we have already surveyed
                        current |= set(self.get_neigh_weights(outer, node_only=True))
                    # frontier only includes nodes we have not surveyed
                    frontier = current - neighbors
                    neighbors |= current
                if CACHING:
                    neighbor_cache.put(node, neighbors)
            far_nodes = homo_nodes - neighbors
            neighbors -= {node}

            # Update positive samples
            pos_samples = random.sample(neighbors, num_pos) if num_pos < len(neighbors) else neighbors
            pos_pairs = [(node, pos_node) for pos_node in pos_samples]
            self.positive_pairs.extend(pos_pairs)
            self.node_positive_pairs[node] = pos_pairs

            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            neg_pairs = [(node, neg_node) for neg_node in neg_samples]
            self.negative_pairs.extend(neg_pairs)
            self.node_negative_pairs[node] = neg_pairs

            if LOGGING and random.random() <= LOG_PROB:
                print("Positive samples of {} is ".format(self.entities[node]),
                      [self.entities[x[1]] for x in pos_pairs])
                print("Negative samples of {} is ".format(self.entities[node]),
                      [self.entities[x[1]] for x in neg_pairs])


    def get_neigh_weights(self, node, node_only=False):
        neighs = self.adj_lists[int(node)]
        if node_only:
            return [int(x.split("#")[0]) for x in neighs]
        neigh_nodes, neigh_weights = [], []
        for x in neighs:
            x = x.split("#")
            neigh_nodes.append(int(x[0]))
            neigh_weights.append(float(x[1]))
        neigh_weights = special.softmax(neigh_weights)
        return neigh_nodes, neigh_weights

    def get_positive_nodes(self, nodes, n_walks, walk_len):
        return self._run_random_walks(nodes, n_walks, walk_len)

    def get_homo_nodes(self, node):
        if node in self.users:
            return self.users - {node}
        return self.news | self.sources - {node}

    def negative_sample(self, node, far_nodes, num_neg):
        neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
        return neg_samples

    def get_negative_nodes(self, nodes, num_neg, neg_walk_len):
        for node in nodes:
            if node in self.far_node_cache:
                far_nodes = self.far_node_cache[node]
            else:
                neighbors, frontier = {node}, {node}
                for _ in range(neg_walk_len):
                    current = set()
                    for outer in frontier:
                        current |= set(self.get_neigh_weights(outer, node_only=True))
                        # Begin: uncomment the code below to sub-sample neighbors
                        # neighs = self.get_neigh_weights(outer, node_only=True)
                        # if len(neighs) > 0:
                        #     neighs = np.random.choice(neighs, 5)
                        # current |= set(neighs)
                        # End
                    frontier = current - neighbors
                    neighbors |= current
                homo_nodes = self.get_homo_nodes(node)
                far_nodes = homo_nodes - neighbors
                self.far_node_cache[node] = far_nodes
            neg_samples = self.negative_sample(node, far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes

            if LOGGING and random.random() <= LOG_PROB:
                print("Negative samples of {} is ".format(self.entities[node]),
                      [self.entities[x] for x in neg_samples])

            self.negative_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negative_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negative_pairs

    def _run_random_walks(self, nodes, n_walks, walk_len):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                self.node_positive_pairs[node] = []
                continue
            curr_pairs = []
            for i in range(n_walks):
                curr_node = node
                for j in range(walk_len):
                    neigh_nodes, neigh_weights = self.get_neigh_weights(curr_node)
                    next_node = np.random.choice(neigh_nodes, 1, p=neigh_weights)[0]
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.sampled_nodes:
                        self.positive_pairs.append((node, next_node))
                        curr_pairs.append((node, next_node))
                    curr_node = next_node

            if LOGGING and random.random() <= LOG_PROB:
                print("Positive samples of {} is ".format(self.entities[node]),
                      [self.entities[x[1]] for x in curr_pairs])

            self.node_positive_pairs[node] = curr_pairs
        return self.positive_pairs
