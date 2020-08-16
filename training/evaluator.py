import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.special import softmax

class Evaluator(object):
    def __init__(self, logits, labels, average="macro"):
        self.logits = logits
        self.predictions = np.argmax(logits, axis=1) if logits is not None else None
        self.labels = labels
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.auc = None
        self.average = average

    def is_better_than(self, evaluator, metrics):
        if "auc" in metrics and self.auc is not None and evaluator.auc is not None \
                and evaluator.auc > self.auc:
            return False
        return True

    def evaluate(self, metrics):
        eval_result = dict()
        print("Number of test: {}".format(len(self.predictions)))
        if "accuracy" in metrics:
            print("Accuracy score: {}".format(self._compute_accuracy()))
            eval_result["accuracy"] = np.mean(self._compute_accuracy())
        if "precision" in metrics:
            print("Precision score: {}".format(self._compute_precision()))
            eval_result["precision"] = np.mean(self._compute_precision())
        if "recall" in metrics:
            print("Recall score: {}".format(self._compute_recall()))
            eval_result["recall"] = np.mean(self._compute_recall())
        if "f1" in metrics:
            print("F1 score: {}".format(self._compute_f1()))
            eval_result["f1"] = np.mean(self._compute_f1())
        if "auc" in metrics:
            print("ROC AUC score: {}".format(self._compute_auc()))
            eval_result["auc"] = np.mean(self._compute_auc())
        return eval_result

    def _compute_auc(self):
        prob = softmax(self.logits, axis=-1)
        real_prob = prob[:,1]
        self.auc = roc_auc_score(y_true=self.labels, y_score=real_prob) \
            if self.auc is None else self.auc
        return self.auc

    def _compute_accuracy(self):
        self.accuracy = accuracy_score(y_true=self.labels, y_pred=self.predictions) \
            if self.accuracy is None else self.accuracy
        return self.accuracy

    def _compute_f1(self):
        self.f1 = f1_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.f1 is None else self.f1
        return self.f1

    def _compute_precision(self):
        self.precision = precision_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.precision is None else self.precision
        return self.precision

    def _compute_recall(self):
        self.recall = recall_score(y_true=self.labels, y_pred=self.predictions, average=self.average) \
            if self.recall is None else self.recall
        return self.recall
