from fang.config import FangConfig
import networkx as nx
from collections import defaultdict
from fang.utils import *
import random
from pprint import pprint


def print_graph_stats(gs_data):
    print("# News: {}".format(len(gs_data.news)))
    real_news = [k for k, v in gs_data.news_labels.items() if v == 1]
    fake_news = [k for k, v in gs_data.news_labels.items() if v == 0]
    print("Fake: {}, Real: {}".format(len(fake_news), len(real_news)))
    print("# Sources: {}".format(len(gs_data.sources)))
    print("# Users: {}".format(len(gs_data.users)))
    print("Stats: ")
    pprint(gs_data.stats)


class FangGraphSageDataset(object):
    def __init__(self, fang_config, is_community, inference=False):
        self.name = "fang_graph_sage"
        self.config = fang_config
        self.is_community = is_community
        self.adj_lists, self.edge_list, self.stance_lists = defaultdict(set), [], defaultdict(list)
        self.feature_data = None
        self.n_stances = None
        self.news_labels = None
        self.n_news_labels = None
        self.class2idx = {}
        self.entities, self.node_name_idx_map = None, None
        self.train_idxs, self.dev_idxs, self.test_idxs = [], [], []
        self.news, self.users, self.sources = set(), set(), set()
        self.stats = {}
        self.rep_entities = read_csv(fang_config.rep_entity_path, True, "\t")   \
            if os.path.exists(fang_config.rep_entity_path) else None
        self.load(inference)

    def load_and_update_adj_lists(self, edge_file):
        edge_csv_content = utils.read_csv(edge_file, True, delimiter="\t")

        for row in edge_csv_content:
            weight = 1. if len(row) == 2 else float(row[2])
            node_src, node_dest = self.node_name_idx_map[row[0]], self.node_name_idx_map[row[1]]
            self.adj_lists[node_src].add(str(node_dest) + NAME_WEIGHT_DELIMITER + str(weight))
            self.adj_lists[node_dest].add(str(node_src) + NAME_WEIGHT_DELIMITER + str(weight))
            self.edge_list.append((node_src, node_dest, weight))

        self.stats[edge_file] = len(edge_csv_content)

    def load_stance_map(self, stance_file):
        stance_content = utils.read_csv(stance_file, True, delimiter="\t")
        stance_map = defaultdict(list)
        for row in stance_content:
            weight = 1. if len(row) == 3 else float(row[2])
            ts_mean = float(row[2]) if len(row) == 3 else float(row[3])
            ts_std = 0. if len(row) == 3 else float(row[4])
            user, news = self.node_name_idx_map[row[0]], self.node_name_idx_map[row[1]]
            stance_map[news].append([user, weight, ts_mean, ts_std])
        for news, engagements in stance_map.items():
            stance_map[news] = sorted(engagements, key=lambda x: x[2])
        self.stats[stance_file] = len(stance_content)
        return stance_map

    def load(self, inference):
        print("Load FANG dataset from {}".format(self.config.root))
        ent_p, feat_p, cite_p, publ_p, rela_p, retw_p, news_info_p, rep_p, sup_p, deny_p, unrelat_p, neu_p, \
            neg_p, neu_sup_p, neg_sup_p = get_default_paths(self.config, self.is_community)
        self.entities = utils.load_text_as_list(ent_p)
        # Build list of train/val/test news indices
        news_label_map = get_news_label_map(news_info_p)
        self.train_idxs, self.dev_idxs, self.test_idxs = get_train_val_test_labels_nodes(self.config, self.entities,
                                                                                         news_label_map, inference)
        # Build graph
        node_names = np.array(self.entities, dtype=np.dtype(str))
        self.node_name_idx_map = {j: i for i, j in enumerate(node_names)}
        for e in self.entities:
            e_idx = self.node_name_idx_map[e]
            if is_tag(NEWS_TAG, e):
                self.news.add(e_idx)
            if is_tag(USER_TAG, e):
                self.users.add(e_idx)
            if is_tag(SOURCE_TAG, e):
                self.sources.add(e_idx)
        # Build news labels
        news_label_map = {self.node_name_idx_map[k]: v for k, v in news_label_map.items()}
        self.news_labels, self.n_news_labels, self.class2idx = encode_class_idx_label(news_label_map)

        feature_content = utils.read_csv(feat_p, True, delimiter="\t")
        feature_data = [[float(x) for x in row[1:]] for row in feature_content]
        self.feature_data = row_normalize(np.asarray(feature_data))

        if FangConfig.CITATION in self.config.relations:
            self.load_and_update_adj_lists(cite_p)
        if FangConfig.PUBLICATION in self.config.relations:
            self.load_and_update_adj_lists(publ_p)
        if FangConfig.RELATIONSHIP in self.config.relations:
            self.load_and_update_adj_lists(rela_p)
        if FangConfig.RETWEET in self.config.relations:
            self.load_and_update_adj_lists(retw_p)
        all_stance_maps = []
        if FangConfig.SUPPORT in self.config.relations:
            all_stance_maps.append(self.load_stance_map(sup_p))
        if FangConfig.DENY in self.config.relations:
            all_stance_maps.append(self.load_stance_map(deny_p))
        if FangConfig.REPORT in self.config.relations:
            all_stance_maps.append(self.load_stance_map(rep_p))
        if FangConfig.COMMENT_NEGATIVE in self.config.relations:
            all_stance_maps.append(self.load_stance_map(neg_p))
        if FangConfig.COMMENT_NEUTRAL in self.config.relations:
            all_stance_maps.append(self.load_stance_map(neu_p))
        if FangConfig.SUPPORT_NEUTRAL in self.config.relations:
            all_stance_maps.append(self.load_stance_map(neu_sup_p))
        if FangConfig.SUPPORT_NEGATIVE in self.config.relations:
            all_stance_maps.append(self.load_stance_map(neg_sup_p))
        all_engaged_news = set()
        self.n_stances = len(all_stance_maps)
        for stance_map in all_stance_maps:
            all_engaged_news |= set(stance_map.keys())
        news_user_stance_map = {}
        for news in all_engaged_news:
            news_user_stance_map[news] = {}
            for i, stance_map in enumerate(all_stance_maps):
                if news in stance_map:
                    engaging_users = stance_map[news]
                    for user, num_stances, ts_mean, ts_std in engaging_users:
                        if user not in news_user_stance_map[news]:
                            news_user_stance_map[news][user] = [list(np.zeros(self.n_stances)), 0, 0]
                        news_user_stance_map[news][user][0][i] = num_stances
                        news_user_stance_map[news][user][1] = ts_mean
                        news_user_stance_map[news][user][2] = ts_std
        for news in news_user_stance_map.keys():
            for user in news_user_stance_map[news]:
                self.stance_lists[news].append((user, news_user_stance_map[news][user]))
                self.stance_lists[user].append((news, news_user_stance_map[news][user]))
                self.edge_list.append((user, news, news_user_stance_map[news][user]))


class FangBotDataset(FangGraphSageDataset):
    def __init__(self, fang_config):
        super(FangBotDataset, self).__init__(fang_config, False, False)

    def load(self, inference):
        print("Load Fang Bot")
        ent_p, feat_p, cite_p, publ_p, rela_p, retw_p, news_info_p, rep_p, sup_p, deny_p, unrelat_p, neu_p, \
            neg_p = get_default_paths(self.config, self.is_community)
        self.entities = utils.load_text_as_list(ent_p)

        user_info = read_csv(os.path.join(self.config.root, "user_info.tsv"), True, "\t")
        user_label_map = {row[0]: row[1] for row in user_info}
        self.train_idxs, self.dev_idxs, self.test_idxs = get_train_val_test_labels_nodes(self.config, self.entities,
                                                                                         user_label_map, inference)

        # Build graph
        node_names = np.array(self.entities, dtype=np.dtype(str))
        self.node_name_idx_map = {j: i for i, j in enumerate(node_names)}
        for e in self.entities:
            e_idx = self.node_name_idx_map[e]
            self.users.add(e_idx)
        # Build news labels
        user_label_map = {self.node_name_idx_map[k]: v for k, v in user_label_map.items()}
        self.news_labels, self.n_news_labels, self.class2idx = encode_class_idx_label(user_label_map)

        feature_content = utils.read_csv(feat_p, True, delimiter="\t")
        feature_data = [[float(x) for x in row[1:]] for row in feature_content]
        self.feature_data = row_normalize(np.asarray(feature_data))
        self.load_and_update_adj_lists(rela_p)


class FangDataset(object):
    def __init__(self, fang_config, is_community):
        self.name = "fang"
        self.config = fang_config

        self.support_adj = None
        self.deny_adj = None
        self.comment_neutral_adj = None
        self.comment_negative_adj = None
        self.report_adj = None
        self.unrelated_adj = None
        self.citation_adj = None
        self.retweet_adj = None
        self.relationship_adj = None
        self.publication_adj = None
        self.features = None
        self.num_labels = None
        self.labels = None
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.is_community = is_community
        self.load()

    def load(self):
        print("Load FANG dataset from {}".format(self.config.root))
        ent_p, feat_p, cite_p, publ_p, \
            rela_p, retw_p, news_info_p, rep_p, \
            sup_p, deny_p, unrelat_p, neu_p, \
            neg_p, neu_sup_p, neg_sup_p = get_default_paths(self.config, self.is_community)

        idx_features_labels = np.genfromtxt(feat_p,
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)
        entities = utils.load_text_as_list(ent_p)

        # Build list of train/val/test news indices
        news_label_map = get_news_label_map(news_info_p)
        # set a pseudo "true" label for entities that are not news
        news_labels = [news_label_map[e] if e in news_label_map else "real" for i, e in enumerate(entities)]
        idx_train, idx_val, idx_test = get_train_val_test_labels_nodes(self.config, entities, news_label_map)
        print("Train {} Dev {} Test {}".format(len(idx_train), len(idx_val), len(idx_test)))

        # Build graph
        node_names = np.array(entities, dtype=np.dtype(str))
        node_name_idx_map = {j: i for i, j in enumerate(node_names)}
        num_nodes = len(entities)

        self._load_all_relation_matrices(cite_p, publ_p,
                                         rela_p, retw_p,
                                         rep_p, sup_p, deny_p,
                                         unrelat_p, neu_p,
                                         neg_p,
                                         node_name_idx_map, num_nodes)

        normalized_features = row_normalize(features)
        self.features = np.array(normalized_features.todense())
        labels = encode_onehot(news_labels)
        self.num_labels = labels.shape[1]
        self.labels = np.where(labels)[1]

        self.train_mask = sample_mask(idx_train, labels.shape[0])
        self.val_mask = sample_mask(idx_val, labels.shape[0])
        self.test_mask = sample_mask(idx_test, labels.shape[0])

    def _load_all_relation_matrices(self, source_citation_path, source_publication_path,
                                    user_relationship_path, user_retweet_path,
                                    report_stance_path, support_stance_path, deny_stance_path,
                                    unrelated_stance_path, comment_neutral_stance_path,
                                    comment_negative_stance_path,
                                    node_name_idx_map, num_nodes):

        if FangConfig.CITATION in self.config.relations:
            print("Create citation relations")
            source_citation_edges_unordered = np.genfromtxt(source_citation_path, dtype=np.dtype(str))
            self.citation_adj = \
                self._create_symmetric_adj(node_name_idx_map, source_citation_edges_unordered, num_nodes, binary=True)

        if FangConfig.PUBLICATION in self.config.relations:
            print("Create publication relations")
            source_publication_edges_unordered = np.genfromtxt(source_publication_path, dtype=np.dtype(str))
            self.publication_adj = \
                self._create_symmetric_adj(node_name_idx_map, source_publication_edges_unordered, num_nodes, binary=True)

        if FangConfig.RELATIONSHIP in self.config.relations:
            print("Create followership relations")
            user_relationship_edges_unordered = np.genfromtxt(user_relationship_path, dtype=np.dtype(str))
            self.relationship_adj = \
                self._create_symmetric_adj(node_name_idx_map, user_relationship_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.RETWEET in self.config.relations:
            print("Create retweet relations")
            user_retweet_edges_unordered = np.genfromtxt(user_retweet_path, dtype=np.dtype(str))
            self.retweet_adj = \
                self._create_symmetric_adj(node_name_idx_map, user_retweet_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.REPORT in self.config.relations:
            print("Create report relations")
            report_edges_unordered = np.genfromtxt(report_stance_path, dtype=np.dtype(str))
            self.report_adj = \
                self._create_symmetric_adj(node_name_idx_map, report_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.SUPPORT in self.config.relations:
            print("Create support relations")
            support_edges_unordered = np.genfromtxt(support_stance_path, dtype=np.dtype(str))
            self.support_adj = \
                self._create_symmetric_adj(node_name_idx_map, support_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.DENY in self.config.relations:
            print("Create deny relations")
            deny_edges_unordered = np.genfromtxt(deny_stance_path, dtype=np.dtype(str))
            self.deny_adj = \
                self._create_symmetric_adj(node_name_idx_map, deny_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.UNRELATED in self.config.relations:
            print("Create unrelated relations")
            unrelated_edges_unordered = np.genfromtxt(unrelated_stance_path, dtype=np.dtype(str))
            self.unrelated_adj = \
                self._create_symmetric_adj(node_name_idx_map, unrelated_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

        if FangConfig.COMMENT_NEGATIVE in self.config.relations:
            print("Create comment-negative relations")
            comment_negative_edges_unordered = np.genfromtxt(comment_negative_stance_path, dtype=np.dtype(str))
            self.comment_negative_adj = \
                self._create_symmetric_adj(node_name_idx_map, comment_negative_edges_unordered, num_nodes,
                                           binary=(not self.is_community))
        if FangConfig.COMMENT_NEUTRAL in self.config.relations:
            print("Create comment-neutral relations")
            comment_neutral_edges_unordered = np.genfromtxt(comment_neutral_stance_path, dtype=np.dtype(str))
            self.comment_neutral_adj = \
                self._create_symmetric_adj(node_name_idx_map, comment_neutral_edges_unordered, num_nodes,
                                           binary=(not self.is_community))

    @staticmethod
    def _create_symmetric_adj(node_name_idx_map, edges_unordered, num_nodes, binary):
        # flatten edge (node name pairs) and look up for node idx
        weights = edges_unordered[:, 2].astype(np.float32) if not binary else np.ones(edges_unordered.shape[0])
        edges_unordered = edges_unordered[:, :2]
        flattened_mapped_edges = list(map(node_name_idx_map.get, edges_unordered.flatten()))
        flattened_mapped_edges = [c if c is not None else -1 for c in flattened_mapped_edges]
        edges = np.array(flattened_mapped_edges,
                         dtype=np.int32).reshape(edges_unordered.shape)
        edges = edges[(edges >= 0).all(axis=1)]
        row, col = edges[:, 0], edges[:, 1]
        sparse_adj = sp.coo_matrix((weights, (row, col)),
                                   shape=(num_nodes, num_nodes),
                                   dtype=np.float32)
        # build symmetric adjacency matrix
        symmetric_adj = sparse_adj + sparse_adj.T.multiply(sparse_adj.T > sparse_adj) \
            - sparse_adj.multiply(sparse_adj.T > sparse_adj)
        normalized_adj = row_normalize(symmetric_adj + sp.eye(symmetric_adj.shape[0]))
        normalized_adj = nx.from_scipy_sparse_matrix(normalized_adj, create_using=nx.DiGraph())

        # For debugging
        # adj = self._sparse_mx_to_torch_sparse_tensor(normalized_adj)
        return normalized_adj

    def get_relation_adj(self, relation):
        if relation == FangConfig.SUPPORT:
            return self.support_adj
        if relation == FangConfig.DENY:
            return self.deny_adj
        if relation == FangConfig.COMMENT_NEUTRAL:
            return self.comment_neutral_adj
        if relation == FangConfig.COMMENT_NEGATIVE:
            return self.comment_negative_adj
        if relation == FangConfig.REPORT:
            return self.report_adj
        if relation == FangConfig.UNRELATED:
            return self.unrelated_adj
        if relation == FangConfig.CITATION:
            return self.citation_adj
        if relation == FangConfig.RETWEET:
            return self.retweet_adj
        if relation == FangConfig.RELATIONSHIP:
            return self.relationship_adj
        if relation == FangConfig.PUBLICATION:
            return self.publication_adj
        raise ValueError("Unsupported relation '{}'".format(relation))


def combine_fang_gsds(primary: FangGraphSageDataset, secondary: FangGraphSageDataset, mode="test"):
    # update feature_data, entities and node_name_idx_map
    secondary_new_entity_idxs = [i for i, n in enumerate(secondary.entities) if n not in primary.entities]
    new_entities = [secondary.entities[i] for i in secondary_new_entity_idxs]
    new_features = np.array([secondary.feature_data[i] for i in secondary_new_entity_idxs])
    primary.entities.extend(new_entities)
    primary.feature_data = np.concatenate([primary.feature_data, new_features], axis=0)
    for entity in new_entities:
        new_idx = len(primary.node_name_idx_map)
        primary.node_name_idx_map[entity] = new_idx

    # combine adj_lists
    for node in secondary.adj_lists.keys():
        new_node_idx = primary.node_name_idx_map[secondary.entities[node]]
        adj_nodes = secondary.adj_lists[node]
        for adj_node in adj_nodes:
            info = adj_node.split(NAME_WEIGHT_DELIMITER)
            new_adj_node_idx = primary.node_name_idx_map[secondary.entities[int(info[0])]]
            primary.adj_lists[new_node_idx].add(str(new_adj_node_idx) + NAME_WEIGHT_DELIMITER + info[1])

    # combine stance_lists
    for node, stances in secondary.stance_lists.items():
        new_node_idx = primary.node_name_idx_map[secondary.entities[node]]
        converted_stances = []
        for stance in stances:
            new_stance_node_idx = primary.node_name_idx_map[secondary.entities[stance[0]]]
            converted_stances.append((new_stance_node_idx, stance[1]))
        primary.stance_lists[new_node_idx] = converted_stances

    # update news labels
    new_news_idxs = []
    for news_idx, label in secondary.news_labels.items():
        # get new news_idx
        new_news_idx = primary.node_name_idx_map[secondary.entities[news_idx]]
        primary.news_labels[new_news_idx] = label
        new_news_idxs.append(new_news_idx)

    # update news, users, sources
    primary.news.update(set([primary.node_name_idx_map[secondary.entities[i]] for i in secondary.news]))
    primary.users.update(set([primary.node_name_idx_map[secondary.entities[i]] for i in secondary.users]))
    primary.sources.update(set([primary.node_name_idx_map[secondary.entities[i]] for i in secondary.sources]))

    # if test mode, update primary's test idx to be secondary news id
    if mode == "test":
        primary.test_idxs = new_news_idxs
    elif mode == "finetune_bot":
        train_idxs, dev_idxs, test_idxs = [], [], []
        for x in new_news_idxs:
            if secondary.node_name_idx_map[primary.entities[x]] in secondary.train_idxs:
                train_idxs.append(x)
            elif secondary.node_name_idx_map[primary.entities[x]] in secondary.dev_idxs:
                dev_idxs.append(x)
            elif secondary.node_name_idx_map[primary.entities[x]] in secondary.test_idxs:
                test_idxs.append(x)

        assert len([x for x in train_idxs if x in dev_idxs]) == 0
        assert len([x for x in train_idxs if x in test_idxs]) == 0
        assert len([x for x in test_idxs if x in dev_idxs]) == 0

        primary.train_idxs = train_idxs
        primary.dev_idxs = dev_idxs
        primary.test_idxs = test_idxs

    elif mode == "finetune":
        primary.train_idxs.extend(primary.dev_idxs + primary.test_idxs)
        train_size, dev_size = int(len(new_news_idxs) * 0.4), int(len(new_news_idxs) * 0.2)
        random.shuffle(new_news_idxs)

        # primary.train_idxs.extend(primary.dev_idxs)
        # primary.train_idxs.extend(primary.test_idxs)
        # primary.train_idxs.extend(new_news_idxs[:train_size])

        primary.train_idxs = new_news_idxs[:train_size]
        primary.dev_idxs = new_news_idxs[train_size:(train_size+dev_size)]
        primary.test_idxs = new_news_idxs[(train_size+dev_size):]
        print("Finetuning new model with {} training samples, {} dev samples, {} test samples"
              .format(len(primary.train_idxs), len(primary.dev_idxs),
                      len(primary.test_idxs)))
    elif mode == "combined":
        new_train_idxs = [primary.node_name_idx_map[secondary.entities[i]] for i in secondary.train_idxs]
        new_dev_idxs = [primary.node_name_idx_map[secondary.entities[i]] for i in secondary.dev_idxs]
        new_test_idxs = [primary.node_name_idx_map[secondary.entities[i]] for i in secondary.test_idxs]
        primary.train_idxs.extend(new_train_idxs)
        primary.dev_idxs.extend(new_dev_idxs)
        primary.test_idxs.extend(new_test_idxs)
    else:
        raise ValueError("Unsupported mode {}".format(mode))
    return primary
