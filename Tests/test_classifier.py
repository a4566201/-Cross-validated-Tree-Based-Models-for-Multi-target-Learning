import Trees
import time
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
from sklearn import tree



X, y = make_multilabel_classification(n_samples=1000,random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y)
my_clf = Trees.CartDecisionTreeClassifier(max_depth=2,pred_prob=False)
sk_clf = tree.DecisionTreeClassifier(max_depth=2)
#
start = time.time()
my_clf.fit(X_train, y_train)
print(time.time()-start)
start = time.time()
sk_clf.fit(X_train, y_train)
print(time.time()-start)

my_res = np.array(my_clf.predict(X_test))
sk_res = sk_clf.predict(X_test)
my_clf.print_tree()
tree.plot_tree(sk_clf)
plt.show()