import numpy as np
import copy
from sklearn.datasets import make_regression, make_multilabel_classification
import Trees


class BaseStackedSingleTarget:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.meta_models = []
        self.pred_stage_one_traning = np.array([])
        self.pred_stage_one_prediction = np.array([])
        self.fitted_base_estimators = []
        self.fitted_estimators = []

    def fit(self, X, y):
        self.n_targets = y.shape[1]
        # First training stage
        for i in range(self.n_targets):
            estimator = copy.copy(self.base_estimator)
            estimator.fit(X, y[:, i])
            self.fitted_base_estimators.append(estimator)
            # meta-variable generation
            pred_i = np.array(estimator.predict(X))
            if i == 0:
                self.pred_stage_one_traning = pred_i
            else:
                self.pred_stage_one_traning = np.concatenate((self.pred_stage_one_traning, pred_i), axis=1)
        # Second training stage
        transformed_training_set = np.concatenate((X, self.pred_stage_one_traning), axis=1)
        for i in range(self.n_targets):
            estimator = copy.copy(self.base_estimator)
            estimator.fit(transformed_training_set, y[:, i])
            self.fitted_estimators.append(estimator)

    def predict(self, X_to_predict):
        for i in range(self.n_targets):
            pred_i = self.fitted_base_estimators[i].predict(X_to_predict)
            if i == 0:
                self.pred_stage_one_prediction = pred_i
            else:
                self.pred_stage_one_prediction = np.concatenate((self.pred_stage_one_prediction, pred_i), axis=1)
        # Final prediction
        predictions = list()
        transformed_input_vectors = np.concatenate((X_to_predict, self.pred_stage_one_prediction),axis=1)
        for row in transformed_input_vectors:
            pred_row = [self.fitted_estimators[i].predict([row]) for i in range(self.n_targets)]
            pred_row = np.ravel(pred_row)
            predictions.append(pred_row)
        return np.array(predictions)


class RegressorStackedSingleTarget(BaseStackedSingleTarget):
    def __init__(self,base_estimator=Trees.CartDecisionTreeRegressor(use_pruning=True)):
        super().__init__(base_estimator)


class ClassifierStackedSingleTarget(BaseStackedSingleTarget):
    def __init__(self, base_estimator=Trees.CartDecisionTreeClassifier(use_pruning=True)):
        super().__init__(base_estimator)

if __name__ == '__main__':
    X, y = make_regression(n_samples=10, n_features=2, n_targets=2)
    X,y = make_multilabel_classification(n_samples=100,n_features=2,n_labels=3)
    stacked = ClassifierStackedSingleTarget()
    stacked.fit(X, y)
    pred = stacked.predict(X)
    o=1
    print(pred)



