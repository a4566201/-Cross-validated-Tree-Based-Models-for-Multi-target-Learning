import Trees
import numpy as np
import gb_loss
import copy
from Nodes import *
from sklearn.model_selection import KFold


class BaseGradientBoosting:
    def __init__(self, base_estimator, n_estimators=500, learning_rate=0.001, max_depth=5, min_samples_leaf=1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        if learning_rate < 0:
            raise ValueError("Learning rate must be in the range of [0.0, inf)")
        self.learning_rate = learning_rate
        self.trees = []
        self.base_estimator.max_depth = max_depth
        self.base_estimator.min_samples_leaf = min_samples_leaf
        if self._is_regression():
            self.loss_function = gb_loss.SquaredError()
        else:
            self.loss_function = gb_loss.LogLoss()

    def fit(self, X, y):
        if len(y.shape) == 1:
            self.n_targets = 1
            y = y.reshape(len(y), 1)
        else:
            self.n_targets = y.shape[1]
        y_pred = np.mean(y, axis=0)
        self.F0 = y_pred
        for i in range(self.n_estimators):
            gradient = self.loss_function.gradient(y, y_pred)
            self.base_estimator.fit(X, gradient)
            tree = copy.copy(self.base_estimator)
            self.trees.append(tree)
            y_pred = y_pred + self.learning_rate * np.array(self.base_estimator.predict(X))

    def predict(self, X):
        return self.F0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)

    def _is_regression(self):
        return isinstance(self, (GradientBoostingRegressor, KfoldGradientBoostingRegressor))


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, base_estimator=Trees.CartDecisionTreeRegressor(), n_estimators=500, learning_rate=0.001,
                 max_depth=5, min_samples_leaf=1):
        super().__init__(base_estimator, n_estimators, learning_rate, max_depth, min_samples_leaf)


class GradientBoostingClassifier(BaseGradientBoosting):
    def __init__(self, base_estimator=Trees.CartDecisionTreeRegressor(), n_estimators=500, learning_rate=0.001,
                 max_depth=5, min_samples_leaf=1):
        super().__init__(base_estimator, n_estimators, learning_rate, max_depth, min_samples_leaf)

    def fit(self, X, y):
        if len(y.shape) == 1:
            self.n_targets = 1
            y = y.reshape(-1, 1)
        else:
            self.n_targets = y.shape[1]
        log_of_odds = np.log(y.sum(axis=0) / (y.shape[0] - y.sum(axis=0)))
        self.F0 = log_of_odds
        probabilities = {}
        probabilities[0] = np.full((len(X), self.n_targets), (np.exp(log_of_odds) / (np.exp(log_of_odds) + 1)))
        self.gamma_per_iter = {}
        for i in range(1, self.n_estimators + 1):
            residual = y - probabilities[i - 1]
            tree = copy.copy(self.base_estimator)
            tree.fit(X, residual)
            self.trees.append(tree)
            gamma_leaf, gammas_per_sample = self.gamma_per_leaf(X, tree, probabilities, i)
            self.gamma_per_iter[i] = gamma_leaf
            log_of_odds = log_of_odds + self.learning_rate * gammas_per_sample
            probabilities[i] = np.array([np.exp(odds) / (np.exp(odds) + 1) for odds in log_of_odds])

    def gamma_per_leaf(self, X, fitted_tree, probabilities, iter):
        leaf_indeces = fitted_tree.apply(X)
        flatten_leaf_indeces = np.hstack([val.ravel() for val in leaf_indeces])
        unique_leaf_indices = np.unique(flatten_leaf_indeces)
        gamma_leaf = {}
        for index_leaf in unique_leaf_indices:
            leaf_i = fitted_tree.dic_leaves[index_leaf]
            sum_leaf = leaf_i.prediction * leaf_i.samples
            row_mask = leaf_indeces == index_leaf
            prev_probability = probabilities[iter - 1][row_mask]
            if leaf_i.n_targets > 1:  # One leaf for all targets
                prev_probability = prev_probability.reshape(-1, leaf_i.n_targets)
            denominator = np.sum(prev_probability * (1 - prev_probability), axis=0)
            gamma_leaf[index_leaf] = sum_leaf / denominator
        gammas_per_sample = np.array([[float(gamma_leaf[index_i][i]) if len(gamma_leaf[index_i]) > 1 else float(
            gamma_leaf[index_i]) for i, index_i in enumerate(leaf_indeces[row_index])]
                                      if isinstance(leaf_indeces[row_index], np.ndarray)
                                      else gamma_leaf[leaf_indeces[row_index]] for row_index in range(len(X))])
        return gamma_leaf, gammas_per_sample

    def predict(self, X):
        self.pred_prob = []
        X = np.array(X)
        y_pred = []
        for row in X:
            y_pred_i = copy.copy(self.F0)
            for i, tree in enumerate(self.trees, start=1):
                node = tree.root
                while True:
                    if isinstance(node, Leaf):  # Leaf node
                        y_pred_i += self.learning_rate * self.gamma_per_iter[i][node.id]
                        break
                    elif isinstance(node, list):
                        node = [node_i.get_child(row) for node_i in node]
                        if all(isinstance(node_i, Leaf) for node_i in node):
                            u = np.array([self.gamma_per_iter[i][node_i.id] for node_i in node]).ravel()
                            y_pred_i += np.array(self.learning_rate * np.array(
                                [self.gamma_per_iter[i][node_i.id] for node_i in node]).ravel())
                            break
                    else:
                        node = node.get_child(row)
            y_pred_i = np.exp(y_pred_i) / (np.exp(y_pred_i) + 1)  # Convert to probability
            self.pred_prob.append(copy.copy(y_pred_i))
            y_pred_i = np.round_(y_pred_i).astype(int)  # Convert to 0/1
            y_pred.append(copy.copy(y_pred_i))
        return np.array(y_pred)

    def predict_proba(self, X):
        _ = self.predict(X)
        return np.array(self.pred_prob)


class KfoldGradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, base_estimator=Trees.LooDecisionTreeRegressor(), n_estimators=500, learning_rate=0.001,
                 max_depth=5, min_samples_leaf=1, cv=KFold(n_splits=10, shuffle=True, random_state=0)):
        super().__init__(base_estimator, n_estimators, learning_rate, max_depth, min_samples_leaf)
        self.cv = cv


class KfoldGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self, base_estimator=Trees.LooDecisionTreeRegressor(), n_estimators=500, learning_rate=0.001,
                 max_depth=5, min_samples_leaf=1, cv=KFold(n_splits=10, shuffle=True, random_state=0)):
        super().__init__(base_estimator, n_estimators, learning_rate, max_depth, min_samples_leaf)
        self.cv = cv
