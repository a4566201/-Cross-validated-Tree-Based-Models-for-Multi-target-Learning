import numpy as np
from utils import *


class Criterion:
    def __init__(self, X, y, column_dtypes, min_samples_leaf):
        if y.ndim == 1:
            self.n_targets = 1
        else:
            self.n_targets = y.shape[1]
        self.n_features = X.shape[1]
        self.column_dtypes = column_dtypes
        self.min_samples_leaf = min_samples_leaf

    def update(self, dataset, n_targets):  # for_ind for the case of LOO model test for individual model
        self.dataset = dataset
        self.n_targets = n_targets
        self.X = self.dataset[:, :self.n_features]
        self.y = self.dataset[:, self.n_features:]
        self.n_samples = self.dataset.shape[0]

    def node_impurity(self):  # Method which will calculate the node impurity
        pass

    def get_split(self, x_i):
        pass


class SSR(Criterion):
    def __init__(self, X, y, column_dtypes, min_samples_leaf):
        super().__init__(X, y, column_dtypes, min_samples_leaf)

    def update(self, dataset, for_ind=False):
        super().update(dataset, for_ind)
        self.y_sq = self.y ** 2

    def get_split(self, x_i):
        best_SSR, index_to_split = np.inf, -1
        cum_sum_sq_left, cum_sum_left, cum_sum_right, cum_sum_sq_right = np.zeros(self.n_targets), np.zeros(
            self.n_targets), self.y.sum(axis=0), self.y_sq.sum(axis=0)
        if self.column_dtypes[x_i] in ('float64', 'int64', 'float32', 'int32'):  # x_i is a numerical feature
            sorted_dataset = self.dataset[self.dataset[:, x_i].argsort()]
            y = sorted_dataset[:, self.n_features:]
            for i in range(self.n_samples - 1):
                # updates  left side
                cum_sum_left += y[i, :]
                cum_sum_sq_left += y[i, :] ** 2
                left_mean = cum_sum_left / (i + 1)
                # updates right side
                cum_sum_right -= y[i, :]
                cum_sum_sq_right -= y[i, :] ** 2
                right_mean = cum_sum_right / (self.n_samples - i - 1)
                if sorted_dataset[i, x_i] == sorted_dataset[i + 1, x_i]:
                    continue
                # calculate left_ssr and right_ssr
                left_ssr = cum_sum_sq_left - (i + 1) * left_mean ** 2
                right_ssr = cum_sum_sq_right - (self.n_samples - i - 1) * right_mean ** 2
                ssr = left_ssr.sum() + right_ssr.sum()
                if min(i + 1, self.n_samples - (i + 1)) >= self.min_samples_leaf:
                    if ssr < best_SSR:
                        best_SSR, index_to_split = ssr, i

        else:
            lst_categorical_values = sorted_categorical(self.dataset, x_i, n_targets=self.n_targets)
            for i in range(len(lst_categorical_values) - 1, 0,
                           -1):  # Run over lst_categorical_values and in each loop remove categorical value
                lst_categorical_values.pop(i)
                left, right = test_split_categorical(self.dataset, x_i, lst_categorical_values)
                ssr = calculate_ssr(left, self.n_targets) + calculate_ssr(right, self.n_targets)
                if min(left_ssr.shape[0], right_ssr.shape[0]) < self.min_samples_leaf:
                    if ssr < best_SSR:
                        best_SSR, index_to_split = ssr, lst_categorical_values
        return best_SSR, index_to_split


class Gini(Criterion):
    def __init__(self, X, y, column_dtypes, min_samples_leaf):
        super().__init__(X, y, column_dtypes, min_samples_leaf)

    def update(self, dataset, for_ind=False):
        super().update(dataset, for_ind)
        self.unique_per_target = [set(self.y[:, i]) for i in
                                  range(self.n_targets)]  # Create list of the unique values for each target
        self.total_values = []  # Will save the total number of time each class appear in each target
        for i in range(self.n_targets):
            unique, counts = np.unique(self.y[:, i], return_counts=True)
            self.total_values.append(dict(zip(unique, counts)))

    def get_split(self, x_i):
        best_gini, index_to_split = np.inf, -1
        self.count_values = [dict.fromkeys(self.unique_per_target[i], 0) for i in
                             range(self.n_targets)]  # count the number of time each class appeared utilize to 0
        if self.column_dtypes[x_i] in ('float64', 'int64', 'float32', 'int32'):  # x_i is a numerical feature
            sorted_dataset = self.dataset[self.dataset[:, x_i].argsort()]
            y = sorted_dataset[:, self.n_features:]
            for i in range(self.n_samples - 1):
                p_l, p_r = 0, 0
                for j in range(self.n_targets):
                    # unique_values = self.unique_per_target[j] # unique values for the j target
                    for value in self.count_values[j]:
                        val = value
                        y_ij = y[i, j]
                        if y[i, j] == value:
                            self.count_values[j][value] = self.count_values[j].get(
                                value) + 1  # increment dictionary value
                            break
                    p_l += 1 - np.square(np.array(list(self.count_values[j].values())) / (i + 1)).sum()
                    count_r_side = np.array(list(self.total_values[j].values())) - np.array(
                        list(self.count_values[j].values()))  # # count_total - count_left_side
                    p_r += 1 - np.square(count_r_side / (self.n_samples - i - 1)).sum()
                weighted_gini = ((i + 1) / self.n_samples) * p_l + ((self.n_samples - i - 1) / self.n_samples) * p_r
                if weighted_gini < best_gini:
                    best_gini, index_to_split = weighted_gini, i
        else:  # x_i is a categorical feature
            lst_categorical_values = sorted_categorical(self.dataset, x_i, n_targets=self.n_targets)
            for i in range(len(lst_categorical_values) - 1, 0,
                           -1):  # Run over lst_categorical_values and in each loop remove categorical value
                lst_categorical_values.pop(i)
                left, right = test_split_categorical(self.dataset, x_i, lst_categorical_values)
                p_left, p_right = left.shape[0] / self.dataset.shape[0], right.shape[0] / self.dataset.shape[0]
                weighted_gini = p_left * calaculate_gini(left, self.n_targets) + p_right * calaculate_gini(right,
                                                                                                           self.n_targets)
                if weighted_gini < best_gini:
                    best_gini, index_to_split = weighted_gini, lst_categorical_values
        return best_gini / self.n_targets, index_to_split
