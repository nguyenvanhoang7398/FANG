from user_embed.graphsage.models import GraphSage
import torch.nn as nn
from scipy import special
import torch
import numpy as np


class FangGraphSage(GraphSage):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device,
                 n_sage_layer=1, gcn=False, agg_func='MEAN'):
        super(FangGraphSage, self).__init__(num_layers, input_size, out_size, raw_features, adj_lists,
                                            device, n_sage_layer, gcn, agg_func)

    def get_neigh_weights(self, node, node_only=False):
        neighs = self.adj_lists[int(node)]
        if node_only:
            return set([int(x.split("#")[0]) for x in neighs])
        neigh_nodes, neigh_weights = [], []
        for x in neighs:
            x = x.split("#")
            neigh_nodes.append(int(x[0]))
            neigh_weights.append(float(x[1]))
        neigh_weights = special.softmax(neigh_weights)
        return neigh_nodes, neigh_weights


class StanceClassifier(nn.Module):
    def __init__(self, embed_dim, n_stances, n_hidden=8):
        super(StanceClassifier, self).__init__()
        self.n_stances = n_stances
        self.n_hidden = n_hidden
        self.user_proj = nn.Linear(embed_dim, n_stances * n_hidden)
        self.news_proj = nn.Linear(embed_dim, n_stances * n_hidden)
        self.norm_coeff = np.sqrt(n_hidden)

    def forward(self, user_embed, news_embed):
        # user_embed, news_embed = (bsz, embed_dim)
        # h_u, h_n = (bsz, n_stances, n_hidden)
        h_u = self.user_proj(user_embed)
        h_n = self.news_proj(news_embed)
        # e = (bsz, n_stances, n_hidden)
        e = (h_u * h_n).view(-1, self.n_stances, self.n_hidden)
        stance_logits = e.sum(-1) / self.norm_coeff     # following self-attention
        return stance_logits


class FakeNewsClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_stances, n_temp_features, n_classes, is_temporal,
                 use_attention, dropout):
        super(FakeNewsClassifier, self).__init__()
        self.is_temporal = is_temporal
        self.use_attention = use_attention
        if is_temporal:
            print("Use temporal")
            self.lstm = nn.LSTM(input_size=embed_dim + n_stances + n_temp_features,
                                hidden_size=hidden_dim,
                                num_layers=2,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=True)
            self.dropout = nn.Dropout(1 - dropout)
            if self.use_attention:
                self.aligned_attn_proj = nn.Linear(in_features=hidden_dim * 2,
                                                   out_features=embed_dim,
                                                   bias=False)
                self.meta_attn_proj = nn.Linear(in_features=n_stances+n_temp_features,
                                                out_features=1,
                                                bias=False)
                last_input_dim = 2 * hidden_dim + embed_dim # (source, news, user engagement)
            else:
                last_input_dim = 4 * hidden_dim + embed_dim
        else:
            print("Do not use temporal")
            last_input_dim = embed_dim
        self.output_layer = nn.Linear(last_input_dim, n_classes)

    def forward(self, news_embed, source_embed, user_embed, stances, timestamps, masked_attn):
        if self.is_temporal:
            if self.use_attention:
                return self.forward_temporal_attention(news_embed, source_embed,
                                                       user_embed, stances, timestamps, masked_attn)
            else:
                return self.forward_temporal(news_embed, source_embed,
                                             user_embed, stances, timestamps, masked_attn)
        else:
            return self.forward_basic(news_embed, source_embed,
                                      user_embed, stances, timestamps, masked_attn)

    def forward_basic(self, news_embed, source_embed, user_embed, stances, timestamps, masked_attn):
        agg_user_emb = torch.sum(user_embed, dim=1)
        return self.output_layer(agg_user_emb), None, agg_user_emb

    def forward_temporal(self, news_embed, source_embed, users_embed, stances, timestamps, masked_attn):
        engage_inputs = torch.cat([users_embed, stances, timestamps], dim=-1)
        _, (h_n, c_n) = self.lstm(engage_inputs)
        eng_out = self.dropout(h_n)
        eng_embed = torch.cat([eng_out[i, :, :] for i in range(eng_out.shape[0])], dim=1)
        combined_news_embed = torch.cat([news_embed, eng_embed], dim=1)
        return self.output_layer(combined_news_embed), None, None

    def forward_temporal_attention(self, news_embed, source_embed, users_embed, stances, timestamps, masked_attn):
        engage_inputs = torch.cat([users_embed, stances, timestamps], dim=-1)
        # hidden_states = (batch_size, seq_len, n_direction * hidden_state)
        hidden_states, _ = self.lstm(engage_inputs)
        # news_embed = (batch_size, embed_size)
        projected_hidden_states = self.aligned_attn_proj(hidden_states)

        # aligned_attentions = (batch_size, seq_len)
        aligned_attentions = torch.bmm(projected_hidden_states, news_embed.unsqueeze(-1)).squeeze(-1)

        meta_inputs = torch.cat([stances, timestamps], dim=-1)
        # meta_attentions = (batch_size, seq_len)
        meta_attentions = self.meta_attn_proj(meta_inputs).squeeze(-1)

        # combine 2 attentions
        aligned_attentions = torch.softmax(aligned_attentions + masked_attn, dim=-1)
        meta_attentions = torch.softmax(meta_attentions + masked_attn, dim=-1)
        attentions = (aligned_attentions + meta_attentions) / 2

        # eng_embed = (batch_size, n_direction * hidden_state)
        eng_embed = torch.bmm(attentions.unsqueeze(1), hidden_states).squeeze(1)

        # compute 1 news representation for fake news, 1 for embedding
        news_embed_rep = torch.add(news_embed, eng_embed)
        news_cls_rep = torch.cat([source_embed, news_embed_rep], dim=1)
        return self.output_layer(news_cls_rep), attentions, news_embed_rep


class NodeClassifier(nn.Module):
    def __init__(self, embed_dim, feature_dim, hidden_dim, n_classes):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim+feature_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, n_classes, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, feature, embed):
        hidden = self.fc1(torch.cat([feature, embed], dim=1))
        relu = self.relu(hidden)
        return self.fc2(relu)


class FeaturelessNodeClassifier(nn.Module):
    def __init__(self, embed_dim, feature_dim, hidden_dim, n_classes):
        super(FeaturelessNodeClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, n_classes, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, feature, embed):
        hidden = self.fc1(embed)
        relu = self.relu(hidden)
        return self.fc2(relu)
