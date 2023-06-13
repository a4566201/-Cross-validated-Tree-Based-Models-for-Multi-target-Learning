import Trees
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt


N_TARGETS = 1
N_SAMPLES = 100
min_samples_leaf = 1
min_samples_leaf = int(0.05*N_SAMPLES)
max_depth = 2

X, y = make_regression(N_SAMPLES,n_features= 10,n_targets=N_TARGETS,random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.5,random_state=0)

sk_regressor = DecisionTreeRegressor(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
sk_regressor.fit(X_train, Y_train)
text_representation = tree.export_text(sk_regressor)
print(text_representation)
sk = pd.DataFrame(sk_regressor.predict(X_test))


my_regressor = Trees.CartDecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
my_regressor.fit(X_train, Y_train)
my_regressor.print_tree()
my =pd.DataFrame(my_regressor.predict(X_test))
tree.plot_tree(sk_regressor)
plt.show()

print(np.round(my) == np.round(sk))

