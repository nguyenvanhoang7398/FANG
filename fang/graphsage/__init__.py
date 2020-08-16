from fang.graphsage.walker import GraphSageWalker
from fang.config import FangConfig
from fang.dataset import FangGraphSageDataset
from fang.graphsage.models import FangGraphSage, StanceClassifier, FakeNewsClassifier
from sklearn.utils import shuffle
from training.configs import BasicConfig
import math
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from training.evaluator import Evaluator
from training.utils import *
from fang.utils import *
from scipy.spatial.distance import cosine
import random
from collections import Counter


AGG_FUNC = "MEAN"
USE_GCN = False
UNSUP_LOSS = "normal"
STANCE_HIDDEN_DIM = 4
N_SAGE_LAYERS = 1
TOP_USER_PER_NEWS = 3
TOP_USER_PER_BATCH = 100

EXPORT_EMBEDDING = True

def load_single(run_config: BasicConfig, fang_config: FangConfig, is_community):
    device = torch.device("cuda:0" if run_config.use_cuda else "cpu")
    data = FangGraphSageDataset(fang_config, is_community)
    meta_data = load_meta(fang_config)

    features = torch.FloatTensor(data.feature_data).to(device)
    embed_size = fang_config.n_hidden
    adj_lists = data.adj_lists  # default dictionary int->set(int) {0: {1, 2}}
    edge_list = data.edge_list
    stance_lists = data.stance_lists
    n_stances = data.n_stances
    n_news_labels = data.n_news_labels
    news_labels = data.news_labels
    dev_nodes, test_nodes = set(data.dev_idxs), set(data.test_idxs)
    news, users, sources = data.news, data.users, data.sources
    n_nodes, input_size = features.size(0), features.size(1)
    return device, data, meta_data, features, embed_size, adj_lists, edge_list, stance_lists, n_stances, n_news_labels, news_labels, \
        dev_nodes, test_nodes, news, users, sources, n_nodes, input_size


def test_fn(run_config, fang_config, data, model_dir, features, n_stances, n_news_labels, input_size, embed_size,
            temporal, attention, device, step, n_nodes, dev_nodes, test_nodes, adj_lists, stance_lists,
            news, users, sources, news_labels):
    graph_sage, stance_classifier, news_classifier = load_save_models(fang_config, data, model_dir, features, n_stances,
                                                                      n_news_labels, input_size, embed_size,
                                                                      temporal, attention, device, step)
    # for train nodes, we add all nodes that are not dev and test nodes
    train_nodes = [x for x in range(n_nodes) if x not in dev_nodes and x not in test_nodes]
    graph_sage_walker = GraphSageWalker(adj_lists, stance_lists, news, users, sources,
                                        (set(train_nodes), dev_nodes, test_nodes),
                                        news_labels, device, max_engages=100)

    test_evaluator, output_attention = eval_fn(test_nodes, graph_sage_walker, graph_sage, news_classifier,
                                               n_stances, embed_size,
                                               run_config.metrics, 0, None, device, tag="test", return_attn=True)


def load_save_models(fang_config, data, model_dir, features, n_stances, n_news_labels, input_size, embed_size,
                     temporal, attention, device, step=-1):
    step = step if step != -1 else "best"
    graph_sage_model_path = os.path.join(model_dir, "graph_sage_ckpt_{}.tar".format(step))
    print("Load model with path {}".format(graph_sage_model_path))
    graph_sage = FangGraphSage(fang_config.n_layers, input_size, embed_size,
                               features, data.adj_lists, device, n_sage_layer=N_SAGE_LAYERS, gcn=USE_GCN,
                               agg_func=AGG_FUNC)
    graph_sage.load_state_dict(torch.load(graph_sage_model_path)["state_dict"])
    graph_sage = graph_sage.to(device)

    stance_classifier_model_path = os.path.join(model_dir, "stance_classifier_ckpt_{}.tar".format(step))
    stance_classifier = StanceClassifier(embed_size, n_stances, STANCE_HIDDEN_DIM)
    stance_classifier.load_state_dict(torch.load(stance_classifier_model_path)["state_dict"])
    stance_classifier = stance_classifier.to(device)

    news_classifier_model_path = os.path.join(model_dir, "news_classifier_ckpt_{}.tar".format(step))
    news_classifier = FakeNewsClassifier(embed_size, int(fang_config.n_hidden/2), n_stances, 2, n_news_labels,
                                         temporal, attention, fang_config.dropout)
    news_classifier.load_state_dict(torch.load(news_classifier_model_path)["state_dict"])
    news_classifier = news_classifier.to(device)
    return graph_sage, stance_classifier, news_classifier


def test_fang_graph_sage(exp_name, model_dir, step, run_config: BasicConfig, fang_config: FangConfig, is_community,
                         temporal, attention, analysis):
    writer = SummaryWriter(os.path.join(run_config.log_dir, exp_name))
    device, data, meta_data, features, embed_size, adj_lists, edge_list, stance_lists, n_stances, n_news_labels, news_labels, \
        dev_nodes, test_nodes, news, users, sources, n_nodes, input_size = \
        load_single(run_config, fang_config, is_community)
    graph_sage, stance_classifier, news_classifier = load_save_models(fang_config, data, model_dir, features, n_stances, n_news_labels, input_size, embed_size,
        temporal, attention, device, step)

    # for train nodes, we add all nodes that are not dev and test nodes
    train_nodes = [x for x in range(n_nodes) if x not in dev_nodes and x not in test_nodes]

    graph_sage_walker = GraphSageWalker(adj_lists, stance_lists, news, users, sources,
                                        (set(train_nodes), dev_nodes, test_nodes),
                                        news_labels, device, fang_config.cache_path, data.entities, max_engages=100)

    if analysis == "":
        test_evaluator, output_attention = eval_fn(test_nodes, graph_sage_walker, graph_sage, news_classifier,
                                                   n_stances, embed_size,
                                                   run_config.metrics, 0, None, device, tag="test", return_attn=True)

    elif analysis == "embedding":
        # TODO: Refactor this duplicated codes
        meta_header = meta_data[0]
        meta_body = meta_data[1:]

        def infer_news_emb(_news_batch):
            st = time.time()
            _news_sources_emb_batch, _news_emb_batch, _news_user_emb_batch, _news_stances_tensor, \
            _news_ts_tensor, _news_labels_tensor, _masked_attn_batch, _news_labels_batch, _engage_users_batch = \
                preprocess_news_classification_data(graph_sage, graph_sage_walker, _news_batch, n_stances,
                                                    embed_size, device, return_user_batch=True)
            _news_logit_batch, _news_attention, _news_emb_batch = news_classifier(
                _news_emb_batch, _news_sources_emb_batch, _news_user_emb_batch, _news_stances_tensor,
                _news_ts_tensor, _masked_attn_batch)
            # print("Infer {} news emb for: {:.4f}s".format(len(_news_batch), time.time() - st))
            return _news_logit_batch, _news_attention, _news_emb_batch, _news_labels_tensor, _engage_users_batch

        all_nodes = list(news) + list(sources)
        print("Exporting train embeddings")
        export_batch_size = run_config.batch_size * 8
        all_node_embed, export_buffer = [], []
        all_node_meta = []
        for node in tqdm(all_nodes, desc="Exporting node embedding to Tensorboard"):
            export_buffer.append(node)
            if len(export_buffer) == export_batch_size:
                # split the buffer into news and none news entities
                export_news = [n for n in export_buffer if n in graph_sage_walker.news_labels]
                export_not_news = [n for n in export_buffer if n not in graph_sage_walker.news_labels]
                if len(export_news) > 0:
                    _, _, buffer_news_emb, _, _ = infer_news_emb(
                        export_news)
                    buffer_news_emb = buffer_news_emb.detach().cpu().numpy()
                    all_node_embed.extend(list(buffer_news_emb))
                    all_node_meta.extend([meta_body[node] for node in export_news])
                    del buffer_news_emb
                if len(export_not_news) > 0:
                    buffer_node_emb = graph_sage(export_not_news).detach().cpu().numpy()
                    all_node_embed.extend(list(buffer_node_emb))
                    all_node_meta.extend([meta_body[node] for node in export_not_news])
                    del buffer_node_emb
                export_buffer = []

        # export last buffer
        if len(export_buffer) > 0:
            export_news = [n for n in export_buffer if n in graph_sage_walker.news_labels]
            export_not_news = [n for n in export_buffer if n not in graph_sage_walker.news_labels]
            if len(export_news) > 0:
                _, _, buffer_news_emb, _, _ = infer_news_emb(
                    export_news)
                buffer_news_emb = buffer_news_emb.detach().cpu().numpy()
                all_node_embed.extend(list(buffer_news_emb))
                all_node_meta.extend([meta_body[node] for node in export_news])
            if len(export_not_news) > 0:
                buffer_node_emb = graph_sage(export_not_news).detach().cpu().numpy()
                all_node_embed.extend(list(buffer_node_emb))
                all_node_meta.extend([meta_body[node] for node in export_not_news])

        all_node_embed = np.vstack(all_node_embed)
        writer.add_embedding(all_node_embed, tag='node embedding', metadata=all_node_meta,
                             global_step=0, metadata_header=meta_header)
        del all_node_embed

    elif analysis == "attention":
        test_evaluator, output_attention = eval_fn(news, graph_sage_walker, graph_sage, news_classifier,
                                                   n_stances, embed_size,
                                                   run_config.metrics, 0, None, device, tag="test", return_attn=True)

        predictions = test_evaluator.predictions
        labels = test_evaluator.labels

        cpu_attention = output_attention.detach().cpu().numpy()
        max_duration = 86400 * 14   # 2 weeks
        cutoff_step = 3600 * 12  # cutoff step = 3 hours
        n_bins = int(max_duration / cutoff_step)
        source_batch, news_batch, engage_users_batch, engage_stances_batch, engage_ts_batch, masked_attn_batch, news_labels_batch = \
            graph_sage_walker.fetch_news_classification(test_nodes, n_stances, None, max_duration)
        real_attn_dist, fake_attn_dist = [], []
        for i, engage_ts in tqdm(enumerate(engage_ts_batch), desc="Analyzing attention"):     # iterating through each example
            attn_bin = np.zeros(n_bins)
            non_zero_bins = 0
            for j, ts in enumerate(engage_ts):
                if ts[0] > 0:
                    ts_mean = (ts[0] - 1e-8)*max_duration
                    bin_idx = min(int(ts_mean/cutoff_step), n_bins-1)
                    attn_bin[bin_idx] += cpu_attention[i][j]
                    if ts[0] > 1e-8:
                        non_zero_bins += 1
            attn_bin *= non_zero_bins
            if predictions[i] == 1:
                real_attn_dist.append(attn_bin)
            else:
                fake_attn_dist.append(attn_bin)
        real_attn = np.mean(real_attn_dist, axis=0)
        fake_attn = np.mean(fake_attn_dist, axis=0)
        all_real_attn = [real_attn[0], np.sum(real_attn[1:4]), np.sum(real_attn[4:-1]), real_attn[-1]]
        all_fake_attn = [fake_attn[0], np.sum(fake_attn[1:4]), np.sum(fake_attn[4:-1]), fake_attn[-1]]
        print("First 12h real attention {}".format(all_real_attn[0] / np.sum(all_real_attn)))
        print("12h - 36h real attention {}".format(all_real_attn[1] / np.sum(all_real_attn)))
        print("36h - 2 weeks real attention {}".format(all_real_attn[2] / np.sum(all_real_attn)))
        print("2 weeks+ real attention {}".format(all_real_attn[3] / np.sum(all_real_attn)))
        print("First 12h fake attention {}".format(all_fake_attn[0] / np.sum(all_fake_attn)))
        print("12h - 36h fake attention {}".format(all_fake_attn[1] / np.sum(all_fake_attn)))
        print("36h - 2 weeks fake attention {}".format(all_fake_attn[2] / np.sum(all_fake_attn)))
        print("2 weeks+ fake attention {}".format(all_fake_attn[3] / np.sum(all_fake_attn)))


def run_fang_graph_sage(exp_name, run_config: BasicConfig, fang_config: FangConfig, is_community,
                        temporal, attention, use_stance, use_proximity, pretrained_dir="", pretrained_step=-1):
    print("Using stance" if use_stance else "Not using stance")
    print("Using proximity" if use_proximity else "Not using proximity")
    device, data, meta_data, features, embed_size, adj_lists, edge_list, stance_lists, n_stances, n_news_labels, \
        news_labels, dev_nodes, test_nodes, news, users, sources, n_nodes, input_size = \
        load_single(run_config, fang_config, is_community)
    train_fn(run_config, fang_config, meta_data, exp_name, dev_nodes, test_nodes,
             embed_size, features, data, device, n_stances, n_news_labels,
             adj_lists, stance_lists, news, users, sources, news_labels, temporal, attention, use_stance, use_proximity,
             n_nodes, input_size, pretrained_dir, pretrained_step)


def train_fn(run_config: BasicConfig, fang_config: FangConfig, meta_data, exp_name, dev_nodes, test_nodes,
             embed_size, features, data, device, n_stances, n_news_labels,
             adj_lists, stance_lists, news, users, sources, news_labels, temporal, attention, use_stance, use_proximity,
             n_nodes, input_size, pretrained_dir="", pretrained_step=-1):

    all_nodes = shuffle(list(range(n_nodes)))
    # for train nodes, we add all nodes that are not dev and test nodes
    train_nodes = [x for x in range(n_nodes) if x not in dev_nodes and x not in test_nodes]
    writer = SummaryWriter(os.path.join(run_config.log_dir, exp_name))

    # we first export the node features into projections as a reference point for representation learning
    meta_header = meta_data[0]
    meta_body = meta_data[1:]
    writer.add_embedding(features, tag='node embedding', metadata=meta_data[1:],
                         global_step=0, metadata_header=meta_data[0])

    graph_sage_walker = GraphSageWalker(adj_lists, stance_lists, news, users, sources,
                                        (set(train_nodes), dev_nodes, test_nodes),
                                        news_labels, device, fang_config.cache_path, data.entities)

    best_evaluator, best_output_path_dicts = Evaluator(logits=None, labels=None), {}
    if len(pretrained_dir) != 0:
        print("Load pretrained model")
        graph_sage, stance_classifier, news_classifier = load_save_models(fang_config, data, pretrained_dir, features,
                                                                          n_stances, n_news_labels, input_size,
                                                                          embed_size,
                                                                          temporal, attention, device, pretrained_step)
        validate_evaluator = eval_fn(dev_nodes, graph_sage_walker, graph_sage, news_classifier,
                                     n_stances, embed_size,
                                     run_config.metrics, 0,
                                     writer, device, tag="validate")
        best_evaluator, best_output_path_dicts = eval_and_save({
            "graph_sage": graph_sage,
            "stance_classifier": stance_classifier,
            "news_classifier": news_classifier
        }, exp_name, run_config, 0, validate_evaluator, best_evaluator, best_output_path_dicts)
    else:
        graph_sage = FangGraphSage(fang_config.n_layers, input_size, embed_size,
                                   features, data.adj_lists, device, n_sage_layer=N_SAGE_LAYERS, gcn=USE_GCN,
                                   agg_func=AGG_FUNC)


        stance_classifier = StanceClassifier(embed_size, n_stances, STANCE_HIDDEN_DIM)
        news_classifier = FakeNewsClassifier(embed_size, int(fang_config.n_hidden/2), n_stances, 2, n_news_labels,
                                             temporal, attention, fang_config.dropout)

    def infer_news_emb(_news_batch):
        st = time.time()
        _news_sources_emb_batch, _news_emb_batch, _news_user_emb_batch, _news_stances_tensor, \
        _news_ts_tensor, _news_labels_tensor, _masked_attn_batch, _news_labels_batch, _engage_users_batch = \
            preprocess_news_classification_data(graph_sage, graph_sage_walker, _news_batch, n_stances,
                                                embed_size, device, return_user_batch=True)
        _news_logit_batch, _news_attention, _news_emb_batch = news_classifier(
                    _news_emb_batch, _news_sources_emb_batch, _news_user_emb_batch, _news_stances_tensor,
                    _news_ts_tensor, _masked_attn_batch)
        # print("Infer {} news emb for: {:.4f}s".format(len(_news_batch), time.time() - st))
        return _news_logit_batch, _news_attention, _news_emb_batch, _news_labels_tensor, _engage_users_batch

    def extract_most_attended_users(_news_attn, _engage_users):
        if _news_attn is not None:
            top_attn_users = set()
            _news_attn = _news_attn.detach().cpu().numpy()
            attn_sorted_args = np.argsort(_news_attn)
            top_args = attn_sorted_args[:, -TOP_USER_PER_NEWS:]
            for i, news_engaged_users in enumerate(_engage_users):
                attn_args = top_args[i]
                for idx in attn_args:
                    if idx < len(news_engaged_users):
                        top_attn_users.add(news_engaged_users[idx])
        else:
            all_engaged_users = []
            for i, news_engaged_users in enumerate(_engage_users):
                all_engaged_users.extend(news_engaged_users)
            user_cnt = Counter(all_engaged_users)
            most_common_users = user_cnt.most_common()
            top_attn_users = set([u[0] for u in most_common_users])
        return top_attn_users

    all_models = [graph_sage, stance_classifier, news_classifier]

    params = []
    for model in all_models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    print("Initialize optimizer with weight decay {}".format(run_config.weight_decay))
    optimizer = torch.optim.Adam(params, weight_decay=run_config.weight_decay)
    optimizer.zero_grad()
    for model in all_models:
        model.zero_grad()
        model.to(device)
        model.train()

    b_sz = run_config.batch_size
    unsup_loss = UNSUP_LOSS
    log_every = run_config.inform_every_batch_num
    eval_every_epoch = run_config.eval_every_epoch_num

    global_step, sampling_time, modeling_time = 0, 0., 0.

    _context_losses, _stance_losses, _news_losses = [], [], []

    train_node_set = set(train_nodes)
    assert len([x for x in train_node_set if x in dev_nodes]) == 0
    assert len([x for x in train_node_set if x in test_nodes]) == 0
    assert len([x for x in dev_nodes if x in test_nodes]) == 0
    assert len(train_nodes) + len(dev_nodes) + len(test_nodes) == n_nodes
    print("Train {} Dev {} Test {}".format(len(train_nodes), len(dev_nodes), len(test_nodes)))

    all_nodes = list(news) + list(sources)
    n_nodes = len(all_nodes)

    n_batch_per_epoch = math.ceil(n_nodes / float(b_sz))

    for epoch in tqdm(range(run_config.epoch_num), desc="Training GraphSage"):
        graph_sage.train()
        news_classifier.train()
        random.shuffle(all_nodes)

        """
        due to the large number of users, choosing all users to update would be highly inefficient
        instead, we make use of attention mechanism to help us sample only the "important" users
        1. We will infer news & source embedding
        2. While inferring news & source embedding, we record top k users that are paid most attention
        3. We proceed to infer those users and optimize their context and stance loss
        """

        for batch_idx in tqdm(range(n_batch_per_epoch), desc="Sampling news batches"):
            context_loss, stance_loss, news_loss = None, None, None
            source_news_node_batch = all_nodes[batch_idx * b_sz:(batch_idx + 1) * b_sz]
            # reset walker
            graph_sage_walker.reset()

            # infer the embeddings of user and source nodes
            # user_batch = [n for n in nodes_batch if n in users]
            # user_emb_batch = graph_sage(user_batch) if len(user_batch) > 0 else None
            source_batch = [n for n in source_news_node_batch if n in sources]
            source_emb_batch = graph_sage(source_batch) if len(source_batch) > 0 else None

            # we split the news into train, val-test and extended news so that
            # - train news can be used to detect fake news
            # - train & val-test news can be used to detect stance
            # all news can be used in proximity loss
            news_batch = [n for n in source_news_node_batch if n in news]
            train_news_batch = [n for n in news_batch if n in train_node_set]
            train_news_emb_batch = None
            val_test_news_batch = [n for n in news_batch if n not in train_node_set]
            val_test_news_emb_batch = None

            all_top_attn_users = set()
            # infer the embedding of news nodes
            # please be very careful otherwise the train-test integrity will be lost
            if len(train_news_batch) > 0:
                # prepare the engagement data for aggregating and inferring news embedding
                # this news_classifier serves as both a temporal aggregator and a fake news classifier
                # we don't care about the attention here
                news_logit_batch, train_news_attn, train_news_emb_batch, news_labels_tensor, train_engage_users = \
                    infer_news_emb(train_news_batch)

                train_top_attn_users = extract_most_attended_users(train_news_attn, train_engage_users)
                all_top_attn_users |= train_top_attn_users

                # for train news we also compute the fake news detection loss
                news_loss = graph_sage_walker.compute_news_loss(news_logit_batch, news_labels_tensor)

            if len(val_test_news_batch) > 0:
                # similar to train news but we don't care about the labels here
                _, val_test_news_attn, val_test_news_emb_batch, _, val_test_engage_users = \
                    infer_news_emb(val_test_news_batch)

                val_test_top_attn_users = extract_most_attended_users(val_test_news_attn, val_test_engage_users)
                all_top_attn_users |= val_test_top_attn_users

            if use_stance:
                engaged_news_batch = train_news_batch + val_test_news_batch
                engaged_user_batch, stance_labels_batch, all_n_users = graph_sage_walker.fetch_news_user_stance2(engaged_news_batch)
                if len(engaged_news_batch) > 0:
                    # we concat the train and val-test embeddings in stance detection so we don't have to infer the embedding again
                    if len(train_news_batch) > 0 and len(val_test_news_batch) > 0:
                        _engaged_news_emb_batch = torch.cat([train_news_emb_batch, val_test_news_emb_batch], dim=0)
                    elif len(train_news_batch) > 0:
                        _engaged_news_emb_batch = train_news_emb_batch
                    elif len(val_test_news_batch) > 0:
                        _engaged_news_emb_batch = val_test_news_emb_batch
                    else:
                        raise ValueError("Engaged news batch should not be empty")
                    # we repeat the embedding along axis 0 by the number of engaged users
                    repeats = torch.LongTensor(all_n_users).to(device)
                    engaged_news_emb_batch = torch.repeat_interleave(_engaged_news_emb_batch, repeats, dim=0)
                    engaged_user_embed_batch = graph_sage(engaged_user_batch)

                    # now we compute the stance detection loss
                    stance_logit_batch = stance_classifier(engaged_news_emb_batch, engaged_user_embed_batch)
                    stance_labels_tensor = torch.LongTensor(stance_labels_batch).to(device)
                    stance_loss = graph_sage_walker.compute_stance_loss(stance_logit_batch, stance_labels_tensor)

            if use_proximity:
                # extend node batch with positive and negative sampling
                graph_sage_walker.extend_source_news_nodes(source_news_node_batch)
                extended_news_source_batch = np.asarray(list(graph_sage_walker.get_unique_node_batch()))

                extended_source_batch = [n for n in extended_news_source_batch if n in sources]
                extended_source_emb_batch = graph_sage(extended_source_batch) \
                    if len(extended_source_batch) > 0 else None

                extended_news_batch = [n for n in extended_news_source_batch
                                       if n in graph_sage_walker.news_labels and n not in news_batch]
                extended_news_emb_batch = None

                if len(extended_news_batch) > 0:
                    # similar to train news but we don't care about the labels here
                    _, extended_news_attn, extended_news_emb_batch, _, extended_engage_users = \
                        infer_news_emb(extended_news_batch)

                    extended_top_attn_users = extract_most_attended_users(extended_news_attn, extended_engage_users)
                    all_top_attn_users |= extended_top_attn_users

                # we only choose a subset of users to update based on the attention while inferring news embedding
                # we first extract those users with highest attention
                user_batch = list(all_top_attn_users)
                graph_sage_walker.extend_user_nodes(user_batch)
                user_batch = graph_sage_walker.get_unique_node_batch()
                user_batch = [n for n in user_batch if n in users]
                user_emb_batch = graph_sage(user_batch) if len(user_batch) > 0 else None
                node_embed_batch_list, nodes_batch = [], []
                if user_emb_batch is not None:
                    node_embed_batch_list.append(user_emb_batch)
                    nodes_batch.extend(user_batch)
                if source_emb_batch is not None:
                    node_embed_batch_list.append(source_emb_batch)
                    nodes_batch.extend(source_batch)
                if extended_source_emb_batch is not None:
                    node_embed_batch_list.append(extended_source_emb_batch)
                    nodes_batch.extend(extended_source_batch)
                if train_news_emb_batch is not None:
                    node_embed_batch_list.append(train_news_emb_batch)
                    nodes_batch.extend(train_news_batch)
                if val_test_news_emb_batch is not None:
                    node_embed_batch_list.append(val_test_news_emb_batch)
                    nodes_batch.extend(val_test_news_batch)
                if extended_news_emb_batch is not None:
                    node_embed_batch_list.append(extended_news_emb_batch)
                    nodes_batch.extend(extended_news_batch)
                node_emb_batch = torch.cat(node_embed_batch_list, dim=0)
                # re-order the node batch so that it matches the emb batch

                if unsup_loss == 'margin':
                    context_loss = graph_sage_walker.compute_unsup_loss_margin(node_emb_batch, nodes_batch)
                elif unsup_loss == 'normal':
                    context_loss = graph_sage_walker.compute_unsup_loss_normal(node_emb_batch, nodes_batch)
                else:
                    raise ValueError("Unsupported unsup loss {}".format(unsup_loss))

            total_loss = 0
            if context_loss is None and stance_loss is None and news_loss is None:
                print("Skipping as the batch only has only sources and/or val/test news "
                                 "and no proximity/stance loss is computed")
                continue
            losses = [context_loss, stance_loss, news_loss]
            for loss in losses:
                if loss is not None:
                    total_loss += loss
            total_loss.backward()
            if run_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, run_config.max_grad_norm)

            total_grad = 0
            for name, param in list(filter(lambda p: p[1].grad is not None, graph_sage.named_parameters())):
                param_grad = param.grad.data.norm(2).item()
                total_grad += param_grad

            optimizer.step()

            optimizer.zero_grad()
            for model in all_models:
                model.zero_grad()

            if context_loss is not None:
                _context_losses.append(float(context_loss.detach().cpu()))
            if stance_loss is not None:
                _stance_losses.append(float(stance_loss.detach().cpu()))
            if news_loss is not None:
                _news_losses.append(float(news_loss.detach().cpu()))

            if global_step % log_every == 0:
                context_loss_val = np.mean(_context_losses) if len(_context_losses) > 0 else None
                stance_loss_val = np.mean(_stance_losses) if len(_stance_losses) > 0 else None
                news_loss_val = np.mean(_news_losses) if len(_news_losses) > 0 else None
                _context_losses, _stance_losses, _news_losses = [], [], []

                # Log training results to tensorboard
                train_results = {
                    "grad": total_grad
                }
                if context_loss_val is not None:
                    train_results["context_loss"] = context_loss_val
                if stance_loss_val is not None:
                    train_results["stance_loss"] = stance_loss_val
                if news_loss_val is not None:
                    train_results["news_loss"] = news_loss_val
                writer.add_scalars("train", train_results, global_step)

                # Log training process to CLI
                # print("Visited {}/{}. Training log at {}:".format(len(visited_nodes), n_nodes, global_step))
                print(train_results)

                # try to log the distance between pre-determined closed news
                if data.rep_entities is not None:
                    n_negs = 10
                    rep_entities = data.rep_entities

                    all_rep_entities = set()
                    for row in rep_entities:
                        all_rep_entities.add(row[0])
                        all_rep_entities.add(row[1])
                    all_rep_entities = list(all_rep_entities)
                    rep_entities_idxs = [data.node_name_idx_map[ent] for ent in all_rep_entities]

                    # we infer the embeddings of closed news
                    # this news_classifier serves as both a temporal aggregator and a fake news classifier
                    # we don't care about the attention here

                    news_idxs = [x for x in rep_entities_idxs if x in news]
                    not_news_idxs = [x for x in rep_entities_idxs if x in users or x in sources]
                    news_logit_batch, _, news_emb_batch, _, _ = infer_news_emb(news_idxs)
                    not_news_emb_batch = graph_sage(not_news_idxs)

                    news_embs = news_emb_batch.detach().cpu().numpy()
                    not_news_embs = not_news_emb_batch.detach().cpu().numpy()
                    entity_emb_map = {}

                    for ent_idx, emb in zip(news_idxs, news_embs):
                        entity_emb_map[ent_idx] = emb
                    for ent_idx, emb in zip(not_news_idxs, not_news_embs):
                        entity_emb_map[ent_idx] = emb

                    stance_closed_avg_ratios, stance_far_avg_ratios = [], []
                    other_closed_avg_ratios, other_far_avg_ratios = [], []
                    news_idxs, not_news_idxs = set(news_idxs), set(not_news_idxs)
                    for row in tqdm(rep_entities, desc="Validating rep learning"):
                        ent1_idx = data.node_name_idx_map[row[0]]
                        ent2_idx = data.node_name_idx_map[row[1]]
                        ent1_emb, ent2_emb = entity_emb_map[ent1_idx], \
                                             entity_emb_map[ent2_idx]
                        dist = 1 if not ent1_emb.any() or not ent2_emb.any() else cosine(ent1_emb, ent2_emb)

                        # try to get some negative examples for relative comparison
                        if ent1_idx in news_idxs:
                            other_entities = news_idxs - {ent1_idx}
                        else:
                            other_entities = not_news_idxs - {ent1_idx}
                        other_entities.discard(row[0])
                        neg_entities = random.sample(other_entities, n_negs) if n_negs < len(other_entities) \
                            else other_entities
                        neg_dist_list = []
                        for ent in neg_entities:
                            neg_emb = entity_emb_map[ent]
                            _dist = 1 if not ent1_emb.any() or not neg_emb.any() else cosine(ent1_emb, neg_emb)
                            neg_dist_list.append(_dist)
                        neg_dist = np.mean(neg_dist_list)
                        neg_dist = neg_dist if neg_dist != 0 else 1e-6  # this is to avoid NaN

                        dist_ratio = dist / neg_dist
                        if row[2] == "closed":
                            if row[3] == "stance":
                                stance_closed_avg_ratios.append(dist_ratio)
                            else:
                                other_closed_avg_ratios.append(dist_ratio)
                        elif row[2] == "far":
                            if row[3] == "stance":
                                stance_far_avg_ratios.append(dist_ratio)
                            else:
                                other_far_avg_ratios.append(dist_ratio)

                    stance_avg_closed_ratio = np.mean(stance_closed_avg_ratios) \
                        if len(stance_closed_avg_ratios) > 0 else 0.
                    stance_avg_far_ratio = np.mean(stance_far_avg_ratios) \
                        if len(stance_far_avg_ratios) > 0 else 0.
                    other_avg_closed_ratio = np.mean(other_closed_avg_ratios) \
                        if len(other_closed_avg_ratios) > 0 else 0.
                    other_avg_far_ratio = np.mean(other_far_avg_ratios) \
                        if len(other_far_avg_ratios) > 0 else 0.
                    rep_entity_log = {
                        "stance_avg_closed_ratio": stance_avg_closed_ratio,
                        "stance_avg_far_ratio": stance_avg_far_ratio,
                        "other_avg_closed_ratio": other_avg_closed_ratio,
                        "other_avg_far_ratio": other_avg_far_ratio
                    }
                    writer.add_scalars("train", rep_entity_log, global_step)
                    print(rep_entity_log)
            global_step += 1

        graph_sage_walker.save()

        # Evaluate performance of fake news detection
        if epoch % eval_every_epoch == 0:
            print("Validating")
            validate_evaluator = eval_fn(dev_nodes, graph_sage_walker, graph_sage, news_classifier,
                                         n_stances, embed_size,
                                         run_config.metrics, global_step,
                                         writer, device, tag="validate")
            print("Testing")
            testing_evaluator = eval_fn(test_nodes, graph_sage_walker, graph_sage, news_classifier,
                                         n_stances, embed_size,
                                         run_config.metrics, global_step,
                                         writer, device, tag="test")
            new_best_evaluator, best_output_path_dicts = eval_and_save({
                "graph_sage": graph_sage,
                "stance_classifier": stance_classifier,
                "news_classifier": news_classifier
            }, exp_name, run_config, epoch, validate_evaluator, best_evaluator, best_output_path_dicts)

            if new_best_evaluator != best_evaluator:
                # Log nodes to tensorboard
                if meta_data is not None and EXPORT_EMBEDDING:
                    print("Exporting train embeddings")
                    export_batch_size = run_config.batch_size * 8
                    all_node_embed, export_buffer = [], []
                    all_node_meta = []
                    for node in tqdm(all_nodes, desc="Exporting node embedding to Tensorboard"):
                        export_buffer.append(node)
                        if len(export_buffer) == export_batch_size:
                            # split the buffer into news and none news entities
                            export_news = [n for n in export_buffer if n in graph_sage_walker.news_labels]
                            export_not_news = [n for n in export_buffer if n not in graph_sage_walker.news_labels]
                            if len(export_news) > 0:
                                _, _, buffer_news_emb, _, _ = infer_news_emb(
                                    export_news)
                                buffer_news_emb = buffer_news_emb.detach().cpu().numpy()
                                all_node_embed.extend(list(buffer_news_emb))
                                all_node_meta.extend([meta_body[node] for node in export_news])
                                del buffer_news_emb
                            if len(export_not_news) > 0:
                                buffer_node_emb = graph_sage(export_not_news).detach().cpu().numpy()
                                all_node_embed.extend(list(buffer_node_emb))
                                all_node_meta.extend([meta_body[node] for node in export_not_news])
                                del buffer_node_emb
                            export_buffer = []

                    # export last buffer
                    if len(export_buffer) > 0:
                        export_news = [n for n in export_buffer if n in graph_sage_walker.news_labels]
                        export_not_news = [n for n in export_buffer if n not in graph_sage_walker.news_labels]
                        if len(export_news) > 0:
                            _, _, buffer_news_emb, _, _ = infer_news_emb(
                                export_news)
                            buffer_news_emb = buffer_news_emb.detach().cpu().numpy()
                            all_node_embed.extend(list(buffer_news_emb))
                            all_node_meta.extend([meta_body[node] for node in export_news])
                        if len(export_not_news) > 0:
                            buffer_node_emb = graph_sage(export_not_news).detach().cpu().numpy()
                            all_node_embed.extend(list(buffer_node_emb))
                            all_node_meta.extend([meta_body[node] for node in export_not_news])

                    all_node_embed = np.vstack(all_node_embed)
                    writer.add_embedding(all_node_embed, tag='node embedding', metadata=all_node_meta,
                                         global_step=global_step, metadata_header=meta_header)
                    del all_node_embed
                best_evaluator = new_best_evaluator

    model_dir = os.path.join(run_config.ckpt_dir, exp_name)
    best_graph_sage, best_stance_classifier, best_news_classifier = load_save_models(fang_config, data, model_dir, features, n_stances,
                                                                      n_news_labels, input_size, embed_size,
                                                                      temporal, attention, device, -1)

    eval_fn(test_nodes, graph_sage_walker, best_graph_sage, best_news_classifier, n_stances, embed_size,
            run_config.metrics, global_step, writer, device, tag="test")
    writer.close()


def eval_fn(dev_nodes, graph_sage_walker, graph_sage, news_classifier, n_stances, embed_size,
            metrics, global_step,
            writer, device, tag="validate", cutoff_range=None, return_attn=False):
    graph_sage.eval()
    news_classifier.eval()
    dev_sources_emb_batch, dev_news_emb_batch, dev_engage_user_emb_batch, dev_engage_stances_tensor, \
        dev_engage_ts_tensor, dev_news_labels_tensor, masked_attn_batch, dev_news_labels_batch = \
        preprocess_news_classification_data(graph_sage, graph_sage_walker, dev_nodes, n_stances,
                                            embed_size, device, cutoff_range, test=True)
    # we don't need the news embeddings here
    dev_news_logit_batch, output_attention, _ = news_classifier(dev_news_emb_batch, dev_sources_emb_batch,
                                                             dev_engage_user_emb_batch, dev_engage_stances_tensor,
                                                             dev_engage_ts_tensor, masked_attn_batch)
    dev_news_loss = graph_sage_walker.compute_news_loss(dev_news_logit_batch, dev_news_labels_tensor)

    dev_news_logit = dev_news_logit_batch.detach().cpu().numpy()
    dev_evaluator = Evaluator(dev_news_logit, dev_news_labels_batch)
    validate_result = dev_evaluator.evaluate(metrics)
    validate_result["loss"] = float(dev_news_loss.detach().cpu())
    print(validate_result)

    if writer is not None:
        writer.add_scalars(tag, validate_result, global_step)
    if return_attn:
        return dev_evaluator, output_attention
    return dev_evaluator


def preprocess_news_classification_data(graph_sage, graph_sage_walker, nodes_batch, n_stances, embed_size,
                                        device, cutoff_range=None, return_user_batch=False, test=False):
    source_batch, news_batch, engage_users_batch, engage_stances_batch, engage_ts_batch, \
        masked_attn_batch, news_labels_batch = \
        graph_sage_walker.fetch_news_classification(nodes_batch, n_stances, cutoff_range)
    news_labels_tensor = torch.LongTensor(news_labels_batch).to(device)
    engage_stances_tensor = torch.FloatTensor(engage_stances_batch).to(device)
    engage_ts_tensor = torch.FloatTensor(engage_ts_batch).to(device)
    engage_masked_attn_tensor = torch.FloatTensor(masked_attn_batch).to(device)

    news_emb_batch, sources_emb_batch = graph_sage(news_batch, test=test), graph_sage(source_batch, test=test)
    engage_user_emb_batch = [graph_sage(e_users, test=test) if len(e_users) > 0
                             else torch.zeros(1, embed_size).to(device) for e_users in engage_users_batch]
    for i, user_list in enumerate(engage_user_emb_batch):
        user_list = user_list[:graph_sage_walker.max_engages]
        if len(user_list) < graph_sage_walker.max_engages:
            padding_mtx = torch.zeros((graph_sage_walker.max_engages - len(user_list), embed_size)).to(device)
            engage_user_emb_batch[i] = torch.cat([user_list, padding_mtx], dim=0)
    engage_user_emb_batch = torch.stack(engage_user_emb_batch, dim=0)
    if not return_user_batch:
        return sources_emb_batch, news_emb_batch, engage_user_emb_batch, engage_stances_tensor, \
               engage_ts_tensor, news_labels_tensor, engage_masked_attn_tensor, news_labels_batch
    else:
        return sources_emb_batch, news_emb_batch, engage_user_emb_batch, engage_stances_tensor, \
               engage_ts_tensor, news_labels_tensor, engage_masked_attn_tensor, news_labels_batch, engage_users_batch
