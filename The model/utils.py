import numpy as np
import pandas as pd


def calculate_ssr(dataset, n_targets):
    col_index = dataset.shape[1] - n_targets  # Index of the first column of the target variables
    SSR = np.square(np.subtract(dataset[:, col_index:], np.mean(dataset[:, col_index:], axis=0))).sum(axis=0).sum()
    return SSR


def calaculate_gini(dataset, n_targets):
    col_index = dataset.shape[1] - n_targets  # Index of the first column of the target variables
    n_samples = dataset.shape[0]
    gini = 0
    for i in range(col_index, dataset.shape[1]):
        s = pd.Series(dataset[:, i])
        p_1 = s.mean()  # the observed proportions of “class 1”
        gini += n_samples * p_1 * (1 - p_1)
    return gini


def test_split_numerical(sorted_dataset, xi_to_split,
                         i):  # Split the dataset based on an variable and number of row (i)
    if i == -1:
        return None, None, None
    while sorted_dataset[i, xi_to_split] == sorted_dataset[i + 1, xi_to_split] and i < sorted_dataset.shape[0] - 1:
        if i == sorted_dataset.shape[0] - 2:
            break
        else:
            i += 1
    left, right = sorted_dataset[:i + 1, :], sorted_dataset[i + 1:, :]  # Split to left and right
    return left, right, i  # Return 2 datasets and updated index split


def test_split_categorical(dataset, x_i,
                           lst_categorical_values):  # Split the dataset based on an attribute and categorical values
    left, right = list(), list()
    for row in dataset:
        if row[x_i] in lst_categorical_values:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)


def convert_dtypes_to_lst(X):  # Take features dataframe object and return dtypes in a list
    return list(X.dtypes)


def get_dic_column_names(X):  # Return dictionary with index as a key
    return dict(enumerate(list(X.columns)))


def sorted_categorical(dataset, x_i,
                       n_targets=1):  # return array with the categorical values in feature x_i, sorted by the sum of target
    # # TODO: How to calculate mean targets in MTR?
    df = pd.DataFrame({'categorical_variable': dataset[:, x_i], 'target': dataset[:, dataset.shape[
                                                                                         1] - n_targets]})  # Create DataFrame object from x_i (categorical variable) and target columns
    df['target'] = df[['target']].astype(float)
    df = df.groupby('categorical_variable').agg({'target': 'mean'}).reset_index()
    df = df.sort_values(by='target', ascending=False)  # sort value order by target
    return df['categorical_variable'].tolist()