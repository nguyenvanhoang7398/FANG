import os

class FangConfig(object):

    SUPPORT = "SUPPORT"
    DENY = "DENY"
    COMMENT_NEUTRAL = "COMMENT_NEUTRAL"
    COMMENT_NEGATIVE = "COMMENT_NEGATIVE"
    SUPPORT_NEUTRAL = "SUPPORT_NEUTRAL"
    SUPPORT_NEGATIVE = "SUPPORT_NEGATIVE"
    REPORT = "REPORT"
    UNRELATED = "UNRELATED"

    CITATION = "CITATION"
    RETWEET = "RETWEET"
    RELATIONSHIP = "RELATIONSHIP"
    PUBLICATION = "PUBLICATION"

    def __init__(self, root, train_test_val, meta_data_path, rep_entity_file, cache_path, relations, self_loop,
                 n_hidden, n_layers, dropout, combination, n_heads):
        self.root = root
        self.train_test_val = train_test_val
        self.meta_data_path = meta_data_path
        self.rep_entity_path = os.path.join(root, rep_entity_file)
        self.cache_path = os.path.join(root, cache_path)
        self.relations = relations
        self.self_loop = self_loop
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.combination = combination
        self.n_heads = n_heads

    @staticmethod
    def get_common(root, train_test_val="train_test_90.json"):
        return FangConfig(
            root=root,
            train_test_val=train_test_val,
            meta_data_path="meta_data.tsv",
            cache_path="far_node_cache_{}.pickle",
            rep_entity_file="rep_entities.tsv",
            relations=[
                FangConfig.SUPPORT_NEUTRAL,
                FangConfig.SUPPORT_NEGATIVE,
                FangConfig.REPORT,
                # FangConfig.SUPPORT,
                FangConfig.DENY,
                FangConfig.RELATIONSHIP,
                FangConfig.CITATION,
                FangConfig.PUBLICATION
            ],
            self_loop=False,
            n_hidden=16,
            n_layers=2,
            dropout=0.,
            combination="concat",
            n_heads=2
        )
