import numpy as np
import copy
from sklearn.datasets import make_regression
import Trees
import random

class BaseChain:
    def __init__(self, base_estimator=None, seed=42):
        self.base_estimator = base_estimator
        self.seed = random.seed(seed)
        self.random_chain = None
        self.fitted_estimators = None


    def fit(self, X, y):
        self.n_targets = y.shape[1]
        self.seed
        ordered_targets = list(range(self.n_targets))
        self.random_chain = random.sample(ordered_targets, len(ordered_targets))
        self.fitted_estimators = [None] * self.n_targets
        for i, random_index in enumerate(self.random_chain):
            y_chain = y[:, self.random_chain[:i]]
            transformed_training_set = np.concatenate((X, y_chain), axis=1)
            estimator = copy.copy(self.base_estimator)
            estimator.fit(transformed_training_set, y[:, random_index])
            self.fitted_estimators[random_index] = estimator


    def predict(self, X_to_predict):
        predictions = np.zeros((len(X_to_predict), self.n_targets))
        transformed_input_vectors = X_to_predict
        for i, random_index in enumerate(self.random_chain):
           pred_i = self.fitted_estimators[random_index].predict(transformed_input_vectors)
           predictions[:, random_index] = pred_i
           transformed_input_vectors = np.concatenate((transformed_input_vectors, pred_i), axis=1)
        return predictions

class RegressorChain(BaseChain):
    def __init__(self, base_estimator=Trees.CartDecisionTreeRegressor(use_pruning=True), seed=42):
        super().__init__(base_estimator, seed)


class ClassifierChain(BaseChain):
    def __init__(self, base_estimator=Trees.CartDecisionTreeClassifier(use_pruning=True), seed=42):
        super().__init__(base_estimator, seed)


if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=3, n_targets=3,random_state=42)
    stacked = RegressorChain()
    stacked.fit(X, y)
    pred = stacked.predict(X)
    o=1
    print(pred)
