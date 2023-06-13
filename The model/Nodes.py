from utils import *
import numpy as np


class Node:
    def __init__(self, dataset: np.array, x_i: int = None, left=None,
                 right=None, impurity_for_split=np.inf, depth=0, parent=None, score=None) -> object:
        self.dataset = dataset
        self.samples = dataset.shape[0]
        self.x_i = x_i
        self.left = left
        self.right = right
        self.impurity_for_split = impurity_for_split
        self.depth = depth
        self.parent = parent
        self.score = score  # Only for LooDecisionTreeRegressor

    def __eq__(self, other):
        return (self.dataset == other.dataset).all()


class NumericalNode(Node):
    def __init__(self, dataset, x_i=None, value=None, index=None, left=None, right=None, impurity_for_split=None,
                 depth=0, parent=None, score=None):
        super().__init__(dataset, x_i, left, right, impurity_for_split, depth, parent, score)
        self.index = index
        self.value = value

    def get_child(self, row):
        if row[self.x_i] <= self.value:
            return self.left
        else:
            return self.right


class CategoricalNode(Node):
    def __init__(self, dataset, x_i=None, categorical_values: list = None, left=None, right=None,
                 impurity_for_split_=None, depth=0, parent=None, score=None):
        super().__init__(dataset, x_i, left, right, impurity_for_split_, depth, parent, score)
        self.categorical_values = categorical_values

    def get_child(self, row):
        if row[self.x_i] in self.categorical_values:
            return self.left
        else:
            return self.right


class Leaf:
    def __init__(self, dataset, parent, depth, n_targets, score=None, regression=True, id=-1):
        self.dataset = dataset
        self.id = id
        if regression:
            self.impurity = calculate_ssr(self.dataset, n_targets)
        else:
            self.impurity = calaculate_gini(self.dataset, n_targets)
        self.n_targets = n_targets
        self.depth = depth
        self.parent = parent
        self.prediction = np.mean(self.dataset[:, self.dataset.shape[1] - self.n_targets:], axis=0)
        self.samples = dataset.shape[0]
        self.score = score

    def get_child(self, row):
        return self
