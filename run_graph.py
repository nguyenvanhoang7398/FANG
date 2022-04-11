import argparse
from training.configs import BasicConfig
from training.news_graph import run_news_graph
from training.utils import get_exp_name
from fang.graphsage import run_fang_graph_sage
from fang.config import FangConfig


class GraphConfig(object):
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

    @staticmethod
    def get_common():
        return GraphConfig(
            dataset_root="data"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Learning')
    parser.add_argument('-t', '--task', type=str, default="cora", help='task name')
    parser.add_argument('-m', '--model', type=str, default="gcn", help='model name')
    parser.add_argument('-p', '--path', type=str, default="data/cora/", help='path to dataset')
    parser.add_argument('--percent', type=int, default=90)
    parser.add_argument('--temporal', action="store_true", help='whether to use temporality')
    parser.add_argument('--use-stance', action="store_true", help='whether to use stance')
    parser.add_argument('--use-proximity', action="store_true", help='whether to use proximity')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--attention', action="store_true", help='whether to use attention')
    parser.add_argument('--pretrained_dir', type=str, default="", help='path to pre-trained model directory')
    parser.add_argument('--pretrained_step', type=int, default=-1, help='pre-trained model step')
    parser.add_argument('--use_cpu', action="store_true", help='whether to use CPU instead of CUDA')
    return parser.parse_args()


def news_graph(task, model, graph_config, run_config, percent):
    exp_name = get_exp_name(task + "_" + str(percent), model)
    run_news_graph(
        exp_name=exp_name,
        dataset_name=task,
        dataset_root=graph_config.dataset_root,
        config=run_config,
        model_name=model,
        percent=percent
    )


def fang(args):
    exp_name = get_exp_name(args.task, args.model)
    basic_config = BasicConfig.get_common(
        epochs=args.epochs, use_cuda=not args.use_cpu)
    fang_config = FangConfig.get_common(args.path, train_test_val="train_test_{}.json".format(args.percent))
    if args.model in ["graph_sage"]:
        run_fang_graph_sage(exp_name, basic_config, fang_config, args.community,
                            args.temporal, args.attention, args.use_stance, args.use_proximity,
                            pretrained_dir=args.pretrained_dir, pretrained_step=args.pretrained_step)
    else:
        raise ValueError("Unsupported model {} for FANG".format(args.model))


if __name__ == "__main__":
    p_args = parse_args()
    config = GraphConfig.get_common()
    if p_args.task == "news_graph":
        news_graph(p_args.task, p_args.model, config, BasicConfig.get_news_graph(
            p_args.epochs, use_cuda=not p_args.use_cpu), p_args.percent)
    elif p_args.task == "fang":
        p_args.community = False
        fang(p_args)
    elif p_args.task == "fang_community":
        p_args.community = True
        fang(p_args)
    else:
        raise ValueError("Unrecognized task {}".format(p_args.task))
