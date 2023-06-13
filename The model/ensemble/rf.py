import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import math
import Trees
import copy


class BaseRandomForest:
    def __init__(self, base_estimator, n_features=None, n_estimators=100, max_depth=np.inf, min_samples_leaf=1,
                 random_state=0):
        self.trees_fitted = []
        self.base_estimator = base_estimator
        self.base_estimator.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_leaf = min_samples_leaf
        self.base_estimator.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.select_features_lst = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y.to_numpy()
        random_state = self.random_state
        for i in range(self.n_estimators):
            bootstrap_indices = self._draw_bootsrap(X, random_state=random_state)
            selected_features = self._select_features(X, random_state=random_state)
            self.select_features_lst.append(selected_features)
            X_boot, y_boot = X[np.ix_(bootstrap_indices, selected_features)], y[bootstrap_indices, :]
            X_boot = pd.DataFrame(X_boot)
            X_boot.columns = selected_features  # Using in print tree
            self.base_estimator.fit(X_boot, y_boot)
            tree = copy.deepcopy(self.base_estimator)
            self.trees_fitted.append(tree)
            random_state += 1

    def predict(self, X_to_predict):
        if isinstance(X_to_predict, pd.DataFrame):
            X_to_predict = X_to_predict.to_numpy()
        predictions = list()
        for row in X_to_predict:
            lst_pred_all_trees = []
            for i, tree in enumerate(self.trees_fitted):
                row_selcted_features = row[self.select_features_lst[i]]
                lst_pred_all_trees.append(tree.predict([row_selcted_features]))
            if isinstance(self, (RandomForestRegrresor, KfoldRandomForestRegrresor)):
                pred = np.array(lst_pred_all_trees).mean(axis=0)
            else:
                pred = max(lst_pred_all_trees, key=lst_pred_all_trees.count())
            predictions.append(pred)
        predictions = np.array(predictions)
        if predictions.shape[2] != 1 and predictions.shape[0] != 1:
            predictions = np.squeeze(predictions)
        else:
            predictions = np.ravel(predictions).reshape(predictions.shape[0], predictions.shape[2])
        return np.array(predictions)

    def _draw_bootsrap(self, X, random_state):
        random_instance = np.random.RandomState(random_state)
        sample_indices = random_instance.randint(0, X.shape[0], size=X.shape[0])
        return sample_indices

    def _select_features(self, X, random_state):
        p = self.n_features
        if p is None:
            if isinstance(self, (RandomForestRegrresor, KfoldRandomForestRegrresor)):
                p = int(X.shape[1] / 3)
            else:
                p = int(math.sqrt(X.shape[1]))
        random_instance = np.random.RandomState(random_state)
        feature_indices = random_instance.choice(range(X.shape[1]), size=p, replace=False)
        return feature_indices

    def print_forest(self):
        for i, tree in enumerate(self.trees_fitted, start=1):
            tree.print_tree()


class RandomForestRegrresor(BaseRandomForest):
    def __init__(self, base_estimator=Trees.CartDecisionTreeRegressor(), n_features=None, n_estimators=100,
                 max_depth=np.inf, min_samples_leaf=5, random_state=0):
        super().__init__(base_estimator, n_features, n_estimators, max_depth, min_samples_leaf, random_state)


class RandomForestClassifier(BaseRandomForest):
    def __init__(self, base_estimator=Trees.CartDecisionTreeClassifier(), n_features=None, n_estimators=100,
                 max_depth=np.inf, min_samples_leaf=1, random_state=0):
        super().__init__(base_estimator, n_features, n_estimators, max_depth, min_samples_leaf,
                         random_state)


class KfoldRandomForestRegrresor(BaseRandomForest):
    def __init__(self, base_estimator=Trees.LooDecisionTreeRegressor(), n_features=None, n_estimators=100,
                 max_depth=np.inf,
                 min_samples_leaf=5, random_state=0, cv=KFold(n_splits=10)):
        super().__init__(base_estimator, n_features, n_estimators, max_depth, min_samples_leaf, random_state)
        self.base_estimator.max_depth = max_depth
        self.base_estimator.cv = cv


class KfoldRandomForestClassifier(BaseRandomForest):
    def __init__(self, base_estimator=Trees.LooDecisionTreeClassifier(), n_features=None, n_estimators=100,
                 max_depth=np.inf,
                 min_samples_leaf=1, random_state=0, cv=KFold(n_splits=10)):
        super().__init__(base_estimator, n_features, n_estimators, max_depth, min_samples_leaf, random_state)
        self.base_estimator.max_depth = max_depth
        self.base_estimator = cv
