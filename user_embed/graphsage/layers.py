import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, n_sage_layer, gcn=False):
        super(SageLayer, self).__init__()

        self.gcn = gcn
        self.input_size = input_size if self.gcn else 2 * input_size
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size))
        self.out_size = out_size

        # layers = [nn.Linear(self.input_size, self.out_size)]
        # for i in range(n_sage_layer-1):
        #     layers.append(nn.Linear(self.out_size, self.out_size))
        # self.linear_layers = nn.ModuleList(layers)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.
        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        pre_activated_neurons = self.weight.mm(combined.t())
        combined = torch.tanh(pre_activated_neurons).t()
        return combined
