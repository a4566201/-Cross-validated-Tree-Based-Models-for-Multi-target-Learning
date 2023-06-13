import numpy as np
from Nodes import *
import pandas as pd
from utils import *
from criterion import *
import copy
from sklearn.metrics import hamming_loss, mean_squared_error
from LOO import *

class BaseDecisionTree:

    def __init__(self, criterion=None, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None, part_of_looregressor=False, use_pruning=False, leaf_id=1):
        # TODO: Define types to paramters and functions variable
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split  # not implementation yet
        self.dataset = None
        self.root = None
        self.part_of_looregressor = part_of_looregressor
        self.use_pruning = use_pruning
        self.leaf_id = leaf_id
        self.dic_leaves = {}

    def fit(self, X, y):
        # TODO: option to numpy array as input
        if self.use_pruning:
            regression = True if isinstance(self, CartDecisionTreeRegressor) or isinstance(self,
                                                                                           LooDecisionTreeRegressor) else False
            self.max_depth = self.prune(X, y, regression, maximum_depth=10)
        if type(X) is np.ndarray or type(y) is np.ndarray:
            X, y = pd.DataFrame(X), pd.DataFrame(y)
        self.dic_column_names = get_dic_column_names(X)
        self.column_dtypes = convert_dtypes_to_lst(X)
        if isinstance(y, pd.Series):
            self.n_targets = 1
        else:
            self.n_targets = y.shape[1]
        self.n_features = X.shape[1]
        X, y = X.values, y.values
        self.dataset = np.c_[X, y]  # create one table from features table (X) and targets table  (y)
        if self.criterion == 'ssr':
            self.criterion = SSR(X, y, self.column_dtypes, self.min_samples_leaf)
        elif self.criterion == 'gini':
            self.criterion = Gini(X, y, self.column_dtypes, self.min_samples_leaf)
        self.root = self._get_best_split(self.dataset)
        if self._is_leaf(self.root):
            self.root = Leaf(dataset=self.root.dataset, parent=None, depth=0, n_targets=self.n_targets,id=self.leaf_id)
            self.dic_leaves[self.leaf_id] = self.root
            self.leaf_id += 1
        else:
            self._split(self.root)

    def predict(self, X_to_predict, return_number_of_samples=False):
        if isinstance(X_to_predict, pd.DataFrame):
            X_to_predict = X_to_predict.to_numpy()
        predictions = list()
        for row in X_to_predict:
            predictions.append(self._predict_one_row(row, return_number_of_samples=return_number_of_samples))
        return predictions

    def _predict_one_row(self, row, return_number_of_samples=False):
        node = self.root
        while True:
            if isinstance(node, Leaf):  # Leaf node
                n_predictions = node.prediction  # np.mean(node.dataset[:, node.dataset.shape[1] - self.n_targets:],
                if isinstance(self.criterion, Gini):
                    n_predictions = np.round_(n_predictions).astype(int) if self.pred_prob == False else n_predictions
                if return_number_of_samples:
                    return n_predictions, node.samples
                else:
                    return n_predictions
            else:  # NumericalNode or CategoricalNode
                node = node.get_child(row)

    def _get_best_split(self, dataset):  # Get best split for all dataset. Return Node object
        best_impurity, index_to_split, xi_to_split = np.inf, -1, -1
        self.criterion.update(dataset, self.n_targets)  # pre-calculate for impurity
        if dataset.shape[0] < 2 * self.min_samples_leaf:
            node = self._get_node(dataset, best_impurity, index_to_split, xi_to_split, score=None)
            return node
        for x_i in range(self.n_features):
            impurity_xi, index_to_split_xi = self.criterion.get_split(x_i)
            if impurity_xi < best_impurity:
                best_impurity, index_to_split, xi_to_split = impurity_xi, index_to_split_xi, x_i
        if self.part_of_looregressor:
            loo_sc = LooScore(dataset, self.criterion, 0, self.n_targets, cv=KFold(n_splits=5))
            score = loo_sc.cut_for_individual_score()
        else:
            score = None
        node = self._get_node(dataset, best_impurity, index_to_split, xi_to_split, score=score)
        return node

    def _get_node(self, dataset, best_impurity, index_to_split, xi_to_split, score=None):
        if self.column_dtypes[xi_to_split] in (
                'float64', 'int64', 'float32', 'int32'):  # xi_to_split is a numerical feature
            sorted_dataset = dataset[dataset[:, xi_to_split].argsort()]  # Sorting dataset by x_i variable
            dataset_left, dataset_right, index_to_split = test_split_numerical(sorted_dataset, xi_to_split,
                                                                               index_to_split)  # The new index_to_split for the
            # case when the value is not unique on the
            if (xi_to_split != -1 and best_impurity != np.inf) and sorted_dataset.shape[0] > 1:
                value = (sorted_dataset[index_to_split, xi_to_split] + sorted_dataset[
                    index_to_split + 1, xi_to_split]) / 2
            else:
                value = None
            node = NumericalNode(dataset=dataset, x_i=xi_to_split, value=value,
                                 index=index_to_split, left=dataset_left, right=dataset_right,
                                 score=score)  # create Node object
            if isinstance(self, CartDecisionTreeClassifier):
                node.impurity_for_split, node.impurity = best_impurity, calaculate_gini(dataset,
                                                                                        n_targets=self.n_targets)
            else:
                node.impurity_for_split, node.impurity = best_impurity, calculate_ssr(dataset, n_targets=self.n_targets)
            return node

        else:  # xi_to_split is a categorical feature
            lst_categorical_values = index_to_split
            dataset_left, dataset_right = test_split_categorical(dataset, xi_to_split, lst_categorical_values)
            node = CategoricalNode(dataset=dataset, x_i=xi_to_split, categorical_values=lst_categorical_values,
                                   left=dataset_left,
                                   right=dataset_right, score=score)  # create Node object
            if isinstance(self, CartDecisionTreeClassifier):
                node.impurity_for_split, node.impurity = best_impurity, calaculate_gini(dataset,
                                                                                        n_targets=self.n_targets)
            else:
                node.impurity_for_split, node.impurity = best_impurity, calculate_ssr(dataset, n_targets=self.n_targets)
            return node

    def _split(self, node):
        left_child = self._get_best_split(node.left)  # create left child (Node object)
        node.left = left_child  # Update left to node object (before that it was 2d array)
        if left_child is not None:
            self._update_node_parent(left_child, node)
        right_child = self._get_best_split(node.right)  # create right child (Node object)
        node.right = right_child  # Update right to node object (before that it was 2d array)
        if right_child is not None:
            self._update_node_parent(right_child, node)
        regression = True if isinstance(self, CartDecisionTreeRegressor) or isinstance(self,
                                                                                       LooDecisionTreeRegressor) else False
        if self._is_leaf(left_child):
            node.left = Leaf(dataset=left_child.dataset, parent=node, depth=left_child.depth, n_targets=self.n_targets,
                             score=node.left.score, regression=regression, id=self.leaf_id)
            self.dic_leaves[self.leaf_id] = node.left
            self.leaf_id += 1
        else:
            self._split(left_child)
        if self._is_leaf(right_child):
            node.right = Leaf(dataset=right_child.dataset, parent=node, depth=right_child.depth,
                              n_targets=self.n_targets,
                              score=node.right.score, regression=regression, id=self.leaf_id)
            self.dic_leaves[self.leaf_id] = node.right
            self.leaf_id += 1
        else:
            self._split(right_child)

    def _update_node_parent(self, node, parent):
        if isinstance(node, list):
            for node_i in node:
                node_i.parent = parent
                node_i.depth = parent.depth + 1
        else:
            node.parent = parent
            node.depth = parent.depth + 1

    def _is_leaf(self, node):
        # TODO: Method of node
        is_leaf = False
        if isinstance(node, Node):
            if node.depth == self.max_depth or node.samples < self.min_samples_split:
                return True
            if node.value is None:
                return True
        if self.part_of_looregressor == True and not isinstance(node, list):
            if node.depth != 0 and self.use_pruning == False:  # Not root
                if node.impurity - node.score <= self.min_impurity_decrease:
                    return True
        return is_leaf

    def prune(self, X, y, regression=True, maximum_depth=50, cv=KFold(n_splits=5, shuffle=True, random_state=1)):
        if cv.n_splits >= X.shape[0]:
            cv = LeaveOneOut()
            n_splits = X.shape[0]
        else:
            n_splits = cv.n_splits
        lst_mse = []
        X, y = np.array(X), np.array(y)
        max_depth = 1
        prev_mse, curr_mse = -1, 0
        DT = copy.deepcopy(self)
        DT.use_pruning = False
        while round(prev_mse, 5) != round(curr_mse, 5):
            prev_mse = curr_mse
            curr_mse = 0
            DT.max_depth = max_depth
            for train_index, test_index in cv.split(X):
                X_train, x_test = pd.DataFrame(X[train_index]), X[test_index]
                y_train, y_test = pd.DataFrame(y[train_index]), y[test_index]
                DT.fit(X_train, y_train)
                y_hat = DT.predict(x_test)
                if regression:
                    curr_mse += mean_squared_error(y_test, y_hat)
                else:
                    curr_mse += hamming_loss(y_test, y_hat)
            lst_mse.append(curr_mse / n_splits)
            max_depth += 1
        return np.argmin(lst_mse) + 1

    def _print_info(self, node,
                    width=4):  # Taken from https://github.com/Eligijus112/decision-tree-python/blob/main/RegressionDecisionTree.py
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces
        const = int(node.depth * width ** 1.5)
        spaces = "-" * const

        if node.depth == 0:
            print("Root")
        if not isinstance(node, Leaf):
            x_i_name = self.dic_column_names[node.x_i]
            if isinstance(node, NumericalNode):
                print(
                    f"depth={node.depth}|{spaces} Split rule- x_i to split:{x_i_name}, value to split:{round(node.value, 3)}")
            else:
                print(
                    f"depth={node.depth}|{spaces} Split rule- x_i to split:{node.x_i}, values to split:{node.categorical_values, 3}")

            print(f"{' ' * const}   | Impurity of the node: {round(node.impurity, 2)}")
            print(f"{' ' * const}   | Impurity of the best split: {round(node.impurity_for_split, 2)}")
            print(f"{' ' * const}   | Count of observations in node: {node.dataset.shape[0]}")
            if node.score:
                print(f"{' ' * const}   |LOO score: {node.score}")
        else:  # Leaf node
            print(
                f"depth={node.depth}   leaf node    {' ' * const}   | Count of observations in node: {node.dataset.shape[0]}")
            print(f"       {' ' * const}                   | Prediction of node: {node.prediction.round(3)}")
            print(f"       {' ' * const}                   | Impurity of the node: {round(node.impurity, 2)}")
            print(f'Leaf id: {node.id}')
            if node.score:
                print(f"       {' ' * const}                   | Loo score: {node.score.round(3)}")

    def print_tree(self, node=None):
        if node is None:
            node = self.root
        self._print_info(node)
        if not isinstance(node, Leaf):  # Edge case when root is the root as well (in loo model)
            left_child = node.left
            right_child = node.right
            if not isinstance(left_child, Leaf):
                self.print_tree(node=node.left)
            else:
                self._print_info(node=left_child)
            if not isinstance(right_child, Leaf):
                self.print_tree(node=node.right)
            else:
                self._print_info(node=right_child)

    def apply(self, X):
        row_leafid = []  # Leaf id of sample i
        X = np.array(X)
        for i, row in enumerate(X):
            node = self.root
            while True:
                if isinstance(node, Leaf):  # Leaf node
                    row_leafid.append(node.id)
                    break
                else:  # NumericalNode or CategoricalNode
                    node = node.get_child(row)
        return np.array(row_leafid)



class CartDecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion='ssr', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0,
                 min_impurity_split=None, part_of_looregressor=False, use_pruning=False, leaf_id=1):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         min_impurity_split, part_of_looregressor, use_pruning, leaf_id)


class CartDecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0,
                 min_impurity_split=None, part_of_looregressor=False, use_pruning=False, pred_prob=False, leaf_id=1):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         min_impurity_split, part_of_looregressor, use_pruning, leaf_id)
        self.pred_prob = pred_prob


class LooBaseDecisionTree(BaseDecisionTree):
    # TODO: change criterion name
    def __init__(self, criterion=None, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0,
                 min_impurity_split=None, part_of_looregressor=True, use_pruning=False, leaf_id=1,
                 cv=KFold(n_splits=10, shuffle=True, random_state=1)):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         min_impurity_split, part_of_looregressor, use_pruning, leaf_id)
        self.cv = cv  # Cross validation

    def _predict_one_row(self, row, return_number_of_samples=False):
        node = self.root
        while True:
            if isinstance(node, list):
                node = [node_i.get_child(row) for node_i in node]
                if all(isinstance(node_i, Leaf) for node_i in node):
                    pred = [leaf_i.prediction for leaf_i in node]
                    pred = [item for sublist in pred for item in sublist]  # create flat list
                    if isinstance(self.criterion, Gini):
                        pred = np.round_(pred)
                    return pred
                    break
            elif isinstance(node, Leaf):
                if isinstance(self, LooDecisionTreeClassifier):
                    return np.round_(node.prediction).astype(int) if self.pred_prob == False else node.prediction
                else:
                    return node.prediction
            else:  # NumericalNode or CategoricalNode
                node = node.get_child(row)

    def get_x_i_to_split(self,
                         dataset) -> tuple():  # Return x_i to split (with the best LOO criterion) and how to cut (for all or individual)
        best_score_for_all = np.inf
        xi_to_split = -1
        if dataset.shape[0] > 2:  # If the number of samples is under 3, we can't use leave one out
            # print('start for all score')
            for x_i in range(self.n_features):
                loo_sc = LooScore(dataset, self.criterion, x_i, self.n_targets, cv=self.cv)
                score_for_all_i = loo_sc.cut_for_all_score()
                if score_for_all_i < best_score_for_all:
                    best_score_for_all = score_for_all_i
                    xi_to_split = x_i
            score_for_individual = loo_sc.cut_for_individual_score()  # / (self.n_targets * dataset.shape[0])
            best_score_for_all = best_score_for_all  # / (self.n_targets * dataset.shape[0])
            if best_score_for_all <= score_for_individual:
                how = "for all"
                score = best_score_for_all
            else:
                how = 'for individual'
                score = score_for_individual
        else:  # The number of samples is under 3
            xi_to_split = 0
            how = "for all"
            score = 0
        return xi_to_split, how, score

    def _get_best_split(self, dataset):
        x_i_to_split, how, score = self.get_x_i_to_split(dataset)
        if how == "for all":
            self.criterion.update(dataset, self.n_targets)
            impurity, index_to_split = self.criterion.get_split(x_i_to_split)
            node = self._get_node(dataset, impurity, index_to_split, x_i_to_split, score=score)
            return node
        else:  # how=="for individual" #
            lst_nodes = []  # List of Nodes for each response variable
            for n in range(self.n_targets):
                X = dataset[:, :dataset.shape[1] - self.n_targets]
                y = dataset[:, dataset.shape[1] - self.n_targets + n]
                new_dataset = np.c_[X, y]
                loo_sc = LooScore(new_dataset, self.criterion, 0, 1, cv=self.cv)
                score = loo_sc.cut_for_individual_score()
                node = self._get_node(new_dataset, best_impurity=np.inf, index_to_split=-1, xi_to_split=-1,
                                      score=score)
                lst_nodes.append(node)
            return lst_nodes

    def _split(self, node):
        if isinstance(node, list):
            for i, node_i in enumerate(node):
                X = pd.DataFrame(node_i.dataset[:, :self.n_features])
                y = pd.DataFrame(node_i.dataset[:, -1])
                regression = True if isinstance(self, LooDecisionTreeRegressor) else False
                max_depth = self.max_depth - node_i.depth
                if regression:
                    DT = CartDecisionTreeRegressor(max_depth=max_depth, part_of_looregressor=True, use_pruning=False,
                                                   min_samples_leaf=self.min_samples_leaf, leaf_id=self.leaf_id + 1)
                else:
                    DT = CartDecisionTreeClassifier(max_depth=max_depth, part_of_looregressor=True, use_pruning=False,
                                                    min_samples_leaf=self.min_samples_leaf)
                DT.fit(X, y)
                self.leaf_id += DT.leaf_id + 1
                self.dic_leaves.update(DT.dic_leaves)
                node[i] = DT.root  #
        else:
            super()._split(node)

    def print_tree(self, node=None):
        if node is None:
            node = self.root
        if isinstance(node, list):
            for i, node_i in enumerate(node):
                print(f"{'**' * 30}sub tree number {i} {'**' * 30}")
                super().print_tree(node_i)
        else:
            super().print_tree(node)

    def apply(self, X):
        row_leafid = []  # Leaf id of sample i
        X = np.array(X)
        for i, row in enumerate(X):
            node = self.root
            while True:
                if isinstance(node, list):
                    node = [node_i.get_child(row) for node_i in node]
                    if all(isinstance(node_i, Leaf) for node_i in node):
                        row_leafid.append(np.array([node_i.id for node_i in node]))
                        break
                elif isinstance(node, Leaf):  # Leaf node
                    row_leafid.append(np.array([node.id] * self.n_targets))
                    break
                else:  # NumericalNode or CategoricalNode
                    node = node.get_child(row)
        return np.array(row_leafid)


class LooDecisionTreeRegressor(LooBaseDecisionTree):
    def __init__(self, criterion='ssr', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0,
                 min_impurity_split=None, part_of_looregressor=True, use_pruning=False, leaf_id=1,
                 cv=KFold(n_splits=10)):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         min_impurity_split, part_of_looregressor, use_pruning, leaf_id, cv)


class LooDecisionTreeClassifier(LooBaseDecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0,
                 min_impurity_split=None, part_of_looregressor=True, use_pruning=False, leaf_id=1,
                 cv=KFold(n_splits=10),
                 pred_prob=False):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         min_impurity_split, part_of_looregressor, use_pruning, leaf_id, cv)
        self.pred_prob = pred_prob

