import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut
from criterion import SSR
import Trees


class LooScore:
    def __init__(self, dataset, criterion, x_i, n_targets, cv):
        self.dataset = dataset
        self.criterion = criterion
        self.x_i = x_i
        self.n_targets = n_targets
        self.cv = cv
        self.sorted_dataset = self.dataset[self.dataset[:, self.x_i].argsort()]

    def cut_for_all_score(self):  # loo score for "cut for all" option (only for x_i)
        loo_score = []
        splitter = self.cv  # Create cross validation object
        if isinstance(splitter, KFold):
            if splitter.n_splits > self.dataset.shape[0]:
                splitter = LeaveOneOut()
        cant_loo = self.dataset.shape[0] <= 1
        if cant_loo:  # Cannot perform LeaveOneOut with n_samples<=1
            return 0
        X = self.sorted_dataset[:, self.x_i]
        y = self.sorted_dataset[:, self.dataset.shape[1] - self.n_targets:]
        if isinstance(self.criterion, SSR):
            DT = Trees.CartDecisionTreeRegressor(max_depth=1)  # Create DecisionTreeRegressor object with only 1 depth
        else:
            DT = Trees.CartDecisionTreeClassifier(max_depth=1, pred_prob=True)
        for train_index, test_index in splitter.split(self.dataset):
            X_train, x_test = pd.DataFrame(X[train_index]), X[test_index].reshape(test_index.shape[0], 1)
            y_train, y_test = pd.DataFrame(y[train_index]), y[test_index]
            DT.fit(X_train, y_train)  # Fit the model to find the best split excluding the observation i.
            pred = DT.predict(x_test)
            loo_score.append(mean_squared_error(pred, y_test) * np.array(pred).size)
        return sum(loo_score)

    def cut_for_individual_score(self):
        loo_score = 0
        for n in range(self.n_targets):
            loo_score_yi = []
            X = self.dataset[:, :self.dataset.shape[1] - self.n_targets]
            y = self.dataset[:, self.dataset.shape[1] - self.n_targets + n]  # Take only 1 response variable
            for x_i in range(X.shape[1]):
                new_dataset = np.c_[X, y]
                loo = LooScore(new_dataset, self.criterion, x_i, 1, cv=self.cv)
                loo_score_yi.append(loo.cut_for_all_score())  # loo score only for yi
            loo_score += min(loo_score_yi)
        return loo_score