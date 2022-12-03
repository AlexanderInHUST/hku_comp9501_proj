import pandas as pd
import numpy as np
import time

heart_df = pd.read_csv('data/heart.csv')
feat_labels = heart_df.columns[:-1]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#
x = heart_df.iloc[:, 0:13].values
y = heart_df.iloc[:, 13].values
#
x = normalize(x, axis=0, norm='l2')
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn import metrics

# Perceptron
from sklearn.linear_model import Perceptron

perc = Perceptron()
time_1 = time.time()
perc.fit(x_train, y_train)
time_2 = time.time()
y_pred = perc.predict(x_test)

print('# Perceptron')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Accuracy Score:', metrics.accuracy_score(y_pred, y_test))
print('Time:', (time_2 - time_1) * 1000)

# SVM Classifier
from sklearn import svm

svm = svm.SVC()
time_1 = time.time()
svm.fit(x_train, y_train)
time_2 = time.time()
y_pred = svm.predict(x_test)

print('# SVM')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Accuracy Score:', metrics.accuracy_score(y_pred, y_test))
print('Time:', (time_2 - time_1) * 1000)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=8)
time_1 = time.time()
forest.fit(x_train, y_train)
time_2 = time.time()
y_pred = forest.predict(x_test)

print('# Random Forest Classifier')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Accuracy Score:', metrics.accuracy_score(y_pred, y_test))
print('Time:', (time_2 - time_1) * 1000)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%-*s & %f & \\\\" % (10, feat_labels[indices[f]], importances[indices[f]]))


# extract one single tree
which_tree = 0
best_score = 0
for indx, single_tree in enumerate(forest.estimators_):
    tree_pred = single_tree.predict(x_test)
    tree_score = metrics.accuracy_score(tree_pred, y_test)
    if tree_score > best_score:
        which_tree = indx
        best_score = tree_score
print(which_tree, best_score)

from sklearn.tree import export_graphviz
import pydot

export_graphviz(forest.estimators_[which_tree],
                out_file='tree.dot',
                feature_names = feat_labels,
                class_names = ["N", "Y"],
                rounded = True, proportion = False,
                precision = 2, filled = True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree' + str(which_tree) + '.png')

# 3-layer fc
from keras.models import Sequential
from keras.layers import Dense
from timeit import default_timer as timer
import keras.callbacks


class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.result = 0
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.result += (timer()-self.starttime) * 1000

cb = TimingCallback()

model = Sequential()
model.add(Dense(8, input_dim=13, kernel_initializer='uniform', activation=None))
model.add(Dense(8, kernel_initializer='uniform', activation=None))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

time_1 = time.time()
model.fit(x_train, y_train, epochs=150, batch_size=10, callbacks=[cb])
time_2 = time.time()
y_pred = model.predict(x_test)

print('# 3-layer fc')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
y_result = []
for data in y_pred:
    if data > 0.5:
        y_result.append(1)
    else:
        y_result.append(0)
print('Accuracy Score:', metrics.accuracy_score(y_result, y_test))
print('Time:', cb.result)