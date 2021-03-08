import pandas as pd
import numpy as np
from preprocess import make_boolean_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score
from decisiontree import DecisionTree


# load 'car' dataset
data = pd.read_csv('datasets/car.data', delimiter=',', header=None)
data = data.sample(n=30, random_state=10)


boolean_data = make_boolean_dataset(data)
X = boolean_data[:, :-1]
y = boolean_data[:, -1]
# split dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=42)
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



