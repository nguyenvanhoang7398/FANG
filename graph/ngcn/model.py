import torch
import torch.nn as nn
import torch.nn.functional as F
from graph.gcn.layers import GraphConvolution


class NGCNConfig(object):
    def __init__(self, num_hidden_units, dropout, combine_type):
        self.dropout = dropout
        self.num_hidden_units = num_hidden_units
        self.combine_type = combine_type

    @staticmethod
    def get_common():
        return NGCNConfig(
            num_hidden_units=16,
            dropout=0.3,
            combine_type="concat"
        )


class NGCN(nn.Module):
    def __init__(self, num_features, num_classes, config):
        super(NGCN, self).__init__()

        n_stances = 7
        self.combine_type = config.combine_type
        self.citation_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.relationship_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.publication_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.support_neutral_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.support_negative_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.deny_conv1 = GraphConvolution(num_features, config.num_hidden_units)
        self.report_conv1 = GraphConvolution(num_features, config.num_hidden_units)

        next_input_dim = config.num_hidden_units * n_stances \
            if config.combine_type == "concat" else config.num_hidden_units
        self.citation_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.relationship_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.publication_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.support_neutral_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.support_negative_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.deny_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)
        self.report_conv2 = GraphConvolution(next_input_dim, config.num_hidden_units)

        self.readout_layer = nn.Linear(next_input_dim, num_classes)

        self.dropout = config.dropout

    def forward(self, x, citation_adj, relationship_adj, publication_adj,
                support_neutral_adj, support_negative_adj, deny_adj, report_adj):
        citation_x = F.relu(self.citation_conv1(x, citation_adj))
        relationship_x = F.relu(self.relationship_conv1(x, relationship_adj))
        publication_x = F.relu(self.publication_conv1(x, publication_adj))
        support_neutral_x = F.relu(self.support_neutral_conv1(x, support_neutral_adj))
        support_negative_x = F.relu(self.support_negative_conv1(x, support_negative_adj))
        deny_x = F.relu(self.deny_conv1(x, deny_adj))
        report_x = F.relu(self.report_conv1(x, report_adj))
        relations = [
            citation_x,
            relationship_x,
            publication_x,
            support_neutral_x,
            support_negative_x,
            deny_x,
            report_x
        ]
        if self.combine_type == "concat":
            x = torch.cat(relations, dim=1)
        else:
            x = torch.stack(relations)
            x = x.sum(dim=0)
        x = F.dropout(x, self.dropout, training=self.training)
        citation_x = F.relu(self.citation_conv2(x, citation_adj))
        relationship_x = F.relu(self.relationship_conv2(x, relationship_adj))
        publication_x = F.relu(self.publication_conv2(x, publication_adj))
        support_neutral_x = F.relu(self.support_neutral_conv2(x, support_neutral_adj))
        support_negative_x = F.relu(self.support_neutral_conv2(x, support_negative_adj))
        deny_x = F.relu(self.deny_conv2(x, deny_adj))
        report_x = F.relu(self.report_conv2(x, report_adj))
        relations = [
            citation_x,
            relationship_x,
            publication_x,
            support_neutral_x,
            support_negative_x,
            deny_x,
            report_x
        ]
        if self.combine_type == "concat":
            x = torch.cat(relations, dim=1)
        else:
            x = torch.stack(relations)
            x = x.sum(dim=0)
        last_layer_rep = F.dropout(x, self.dropout, training=self.training)
        x = self.readout_layer(last_layer_rep)
        return F.log_softmax(x, dim=-1), last_layer_rep
