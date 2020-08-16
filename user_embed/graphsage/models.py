from user_embed.graphsage.layers import SageLayer
import torch
import random
import torch.nn as nn


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device,
                 n_sage_layer=1, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.n_sage_layer = n_sage_layer
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.adj_lists = adj_lists

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size,
                                                               n_sage_layer=n_sage_layer, gcn=self.gcn))

    def forward(self, nodes_batch, test=False):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        # nodes_batch = [62640, 8361, 13567]
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # for each layer sample the neighbors of the previous layers
        # nodes_batch_layers is order from deeper layer to shallower layer
        layer_num_samples = [16, 4] if not test else [None, None]  # to avoid OOM we have to restrict the samples of higher level
        # if testing we will not sample neighbor randomly

        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes, num_sample=layer_num_samples[i])
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]   # list of all affected nodes in this layer
            pre_neighs = nodes_batch_layers[index - 1]  # list of all precursor nodes in previous layer
            # aggregate feature of neighbors from previous layer
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer_name = 'sage_layer' + str(index)
            sage_layer = getattr(self, sage_layer_name)
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def get_neigh_weights(self, node, node_only=False):
        return self.adj_lists[int(node)]

    def _get_unique_neighs_list(self, nodes, num_sample=None):
        # for each node get the list of unique neighbors
        to_neighs = [self.get_neigh_weights(int(node), node_only=True) for node in nodes]
        if num_sample is not None:
            # randomly sample num_sample of neighbors
            samp_neighs = [set(random.sample(to_neigh, num_sample))
                           if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        # add the original node into the sampled neighbors
        samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]
        # get the unique node list by adding all sampled neighbors across nodes
        _unique_nodes_list = list(set.union(*samp_neighs)) if len(samp_neighs) > 0 else []
        i = list(range(len(_unique_nodes_list)))
        # construct a node -> unique_id from unique_nodes_list
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        # make sure that the original node is also included in the neighbors of previous layer
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            # remove the original node from the sampled neighbors
            samp_neighs = [(samp_neighs[i] - {nodes[i]}) for i in range(len(samp_neighs))]
        # small gain in efficiency if we check if the unique nodes cover all nodes in graph
        # only extract the embedding of nodes in unique node list
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # embed_matrix is the representation of precursor nodes
        # create a binary mask = (#batch_nodes, #unique_nodes) to indicate which node
        # is the neighbor of each node in batch nodes
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            num_neigh[num_neigh == 0] = 1   # this normalization is to avoid divide by zero
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)
            aggregate_feats[aggregate_feats != aggregate_feats] = 0.

        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        else:
            raise ValueError("Unrecognized aggregating method {}".format(self.agg_func))

        return aggregate_feats
