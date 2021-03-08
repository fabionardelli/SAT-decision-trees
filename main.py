import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score
from decisiontree import DecisionTree


# load 'car' dataset
data = pd.read_csv('datasets/car.data', delimiter=',', header=None)
data = data.sample(n=30, random_state=1)


# preprocess data to get boolean features
# convert a dataset with discrete features into a dataset of boolean features

feature_domains = []  # to store features domains
boolean_col_number = 0  # num. of columns of the new dataset

new_col = 0  # col index of current feature value in 'boolean_data'
for col, feature in data.iteritems():
    current_feature_domain = {}

    # store each feature's value in a dictionary of
    # pairs (value, new_col)
    for value in feature.values:
        if value not in current_feature_domain:
            current_feature_domain[value] = new_col
            new_col += 1

    feature_domains.append(current_feature_domain)
    boolean_col_number += len(current_feature_domain)

# this array will contain the boolean dataset
boolean_data = np.zeros(shape=(data.shape[0], boolean_col_number), dtype=np.int8)

# populate the boolean dataset
for col, feature in data.iteritems():
    for row, value in enumerate(feature):
        boolean_data[row, feature_domains[col][value]] = 1

X = boolean_data[:, :-1]
y = boolean_data[:, -1]
# split dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)
#print(X_train)
#print(y_train)

dt = DecisionTree()
dt.fit_optimal(X_train, y_train)

y_predicted = np.empty(shape=len(y_test), dtype=np.int8)

# collect predictions for each example
for i, example in enumerate(X_test):
    y_predicted[i] = dt.predict(example)

for i, example in enumerate(y_test):
    print(y_test[i], y_predicted[i])

print('Precision:', precision_score(y_test, y_predicted))
print('Recall:', recall_score(y_test, y_predicted))
print('Avg precision:', average_precision_score(y_test, y_predicted))
print('F1:', f1_score(y_test, y_predicted))



