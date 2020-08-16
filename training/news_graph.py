from dataset.news_graph import NewsGraph
from graph.ngcn.model import NGCN, NGCNConfig
import torch.nn.functional as torch_f
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from training.evaluator import Evaluator
from training.utils import *
from training.configs import BasicConfig
from fang.utils import *


def run_news_graph(exp_name, dataset_root, dataset_name, model_name, config: BasicConfig, percent,
                   test_only=False, best_output_path=""):
    news_graph_dataset = NewsGraph(dataset_root, dataset_name, percent)
    meta_data = read_csv(os.path.join(dataset_root, dataset_name, "meta_data.tsv"), True, "\t")

    if model_name == "gcn":
        news_graph_dataset.full_epoch_load(sparse=True)
        model_config = NGCNConfig.get_common()
        model = NGCN(num_features=news_graph_dataset.features.shape[1],
                     num_classes=news_graph_dataset.labels.max().item() + 1,
                     config=model_config)
    else:
        raise ValueError("Unsupported model {} for Graph Learning".format(model_name))

    device = torch.device("cuda:0" if config.use_cuda else "cpu")
    if not test_only:
        # Begin training and validation
        print("Device {}".format(device))
        writer = SummaryWriter(os.path.join(config.log_dir, exp_name))

        best_evaluator, best_output_path = Evaluator(logits=None, labels=None), ""
        model = model.to(device)
        model.zero_grad()

        optimizer = optim.Adam(model.parameters(),
                               weight_decay=config.weight_decay)
        for epoch in range(config.epoch_num):
            loss, train_evaluator, last_layer_rep = train_fn(model, optimizer, news_graph_dataset, device)
            print("Finish training {} epochs, loss: {}.".format(epoch, loss))

            if epoch % config.eval_every_epoch_num == 0:
                train_result = train_evaluator.evaluate(config.metrics)
                train_result["loss"] = loss
                print(train_result)
                writer.add_scalars("train", train_result, epoch)

                if meta_data is not None:
                    print("Log nodes to tensorboard")
                    writer.add_embedding(last_layer_rep, tag='node embedding', metadata=meta_data[1:],
                                         global_step=epoch, metadata_header=meta_data[0])
                print("Validating")
                validate_loss, validate_evaluator = eval_fn(model, news_graph_dataset, device,
                                                            news_graph_dataset.idx_val)
                validate_result = validate_evaluator.evaluate(config.metrics)
                validate_result["loss"] = validate_loss
                print(validate_result)
                writer.add_scalars("validate", validate_result, epoch)
                print("Testing")
                test_loss, test_evaluator = eval_fn(model, news_graph_dataset, device,
                                                            news_graph_dataset.idx_test)
                test_result = test_evaluator.evaluate(config.metrics)
                test_result["loss"] = test_loss
                print(test_result)
                writer.add_scalars("test", test_result, epoch)
                best_evaluator, best_output_path = eval_and_save(model, exp_name, config, epoch, validate_evaluator,
                                                                 best_evaluator, best_output_path)

        print("Training completed, start testing best model from {}.".format(best_output_path))
        model.load_state_dict(torch.load(best_output_path)["state_dict"])
        test_loss, test_evaluator = eval_fn(model, news_graph_dataset, device, news_graph_dataset.idx_test)
        print("Finish testing, loss: {}.".format(test_loss))
        test_result = test_evaluator.evaluate(config.metrics)
        test_result["loss"] = test_loss
        writer.add_scalars("test", test_result, config.epoch_num)
        writer.close()
    else:
        print("Test-only model from {}.".format(best_output_path))
        model.load_state_dict(torch.load(best_output_path)["state_dict"])
        model = model.to(device)
        test_loss, test_evaluator = eval_fn(model, news_graph_dataset, device, news_graph_dataset.idx_test)
        print("Finish testing, loss: {}.".format(test_loss))
        test_result = test_evaluator.evaluate(config.metrics)
    return best_output_path


def train_fn(model, optimizer, news_graph_dataset, device):
    model.train()
    features = news_graph_dataset.features.to(device)
    labels = news_graph_dataset.labels.to(device)
    idx_train = news_graph_dataset.idx_train.to(device)

    citation_adj = news_graph_dataset.citation_adj.to(device)
    relationship_adj = news_graph_dataset.relationship_adj.to(device)
    publication_adj = news_graph_dataset.publication_adj.to(device)
    support_neutral_adj = news_graph_dataset.support_neutral_adj.to(device)
    support_negative_adj = news_graph_dataset.support_negative_adj.to(device)
    deny_adj = news_graph_dataset.deny_adj.to(device)
    report_adj = news_graph_dataset.report_adj.to(device)

    model.train()
    optimizer.zero_grad()

    output, last_layer_rep = model(features, citation_adj, relationship_adj, publication_adj,
                   support_neutral_adj, support_negative_adj, deny_adj, report_adj)
    train_logits = output[idx_train]
    train_labels = labels[idx_train]
    logits = train_logits.detach().cpu().numpy()
    loss_train = torch_f.nll_loss(train_logits, train_labels)
    loss_train.backward()
    optimizer.step()

    train_loss = loss_train.item()
    train_evaluator = Evaluator(logits, train_labels.cpu())

    return train_loss, train_evaluator, last_layer_rep


def eval_fn(model, news_graph_dataset, device, idx_val):
    model.eval()
    features = news_graph_dataset.features.to(device)
    labels = news_graph_dataset.labels.to(device)
    idx_val = idx_val.to(device)

    citation_adj = news_graph_dataset.citation_adj.to(device)
    relationship_adj = news_graph_dataset.relationship_adj.to(device)
    publication_adj = news_graph_dataset.publication_adj.to(device)
    support_neutral_adj = news_graph_dataset.support_neutral_adj.to(device)
    support_negative_adj = news_graph_dataset.support_negative_adj.to(device)
    deny_adj = news_graph_dataset.deny_adj.to(device)
    report_adj = news_graph_dataset.report_adj.to(device)
    
    output, _ = model(features, citation_adj, relationship_adj, publication_adj,
                      support_neutral_adj, support_negative_adj, deny_adj, report_adj)
    val_logits = output[idx_val]
    val_labels = labels[idx_val]
    logits = val_logits.detach().cpu().numpy()
    loss_val = torch_f.nll_loss(val_logits, val_labels)

    val_loss = loss_val.item()
    val_evaluator = Evaluator(logits, val_labels.cpu())

    return val_loss, val_evaluator

