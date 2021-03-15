import pandas as pd
import numpy as np
from preprocess import make_boolean_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score, \
    accuracy_score, matthews_corrcoef
from sklearn import tree
from decisiontree import DecisionTree
import time


def get_mean_scores(res_list):

    num = len(res_list)
    if num == 0:
        raise ValueError("Empty res_list!")

    r = ResultSet()

    for res in res_list:
        r.nodes += res.nodes
        r.time += res.time
        r.precision += res.precision
        r.recall += res.recall
        r.avg_precision += res.avg_precision
        r.f1 += res.f1
        r.accuracy += res.accuracy
        r.matthews += res.matthews

    r.nodes /= num
    r.time /= num
    r.precision /= num
    r.recall /= num
    r.avg_precision /= num
    r.f1 /= num
    r.accuracy /= num
    r.matthews /= num

    return r


# load 'car' dataset
data = pd.read_csv('datasets/car.data', delimiter=',', header=None)
# sample_data = df.sample(n=35, random_state=42)
# test_data = pd.concat([data, df]).drop_duplicates(keep=False)
# select only examples with two class values
# classes = ['unacc', 'acc']
# data = data[data[6].isin(classes)]

# boolean class values
y_data = data[6]
y_data.replace(to_replace=dict(unacc=0, acc=1, good=1, vgood=1), inplace=True)
y = y_data.to_numpy(dtype=np.int8)

# boolean features' values
X_data = data.drop(6, axis=1)
X = make_boolean_dataset(X_data)

dt_results_list = []
scikit_results_list = []


class ResultSet:
    def __init__(self):
        self.nodes = 0
        self.time = 0
        self.precision = 0
        self.recall = 0
        self.avg_precision = 0
        self.f1 = 0
        self.accuracy = 0
        self.matthews = 0


dt = DecisionTree()
scikit_dt = tree.DecisionTreeClassifier()

for s in range(1, 21):
    # split dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1708, random_state=s)

    start = time.time()
    n = dt.fit_optimal(X_train, y_train)
    end = time.time()

    dt_result = ResultSet()

    if n != -1:
        y_predicted = np.empty(shape=len(y_test), dtype=np.int8)

        # collect predictions for each example
        for i, example in enumerate(X_test):
            y_predicted[i] = dt.predict(example)

        dt_result.nodes = n
        dt_result.time = end - start
        dt_result.precision = precision_score(y_test, y_predicted)
        dt_result.recall = recall_score(y_test, y_predicted)
        dt_result.avg_precision = average_precision_score(y_test, y_predicted)
        dt_result.f1 = f1_score(y_test, y_predicted)
        dt_result.accuracy = accuracy_score(y_test, y_predicted)
        dt_result.matthews = matthews_corrcoef(y_test, y_predicted)

        dt_results_list.append(dt_result)

        start = time.time()
        scikit_dt.fit(X_train, y_train)
        end = time.time()

        scikit_result = ResultSet()

        y_predicted = np.empty(shape=len(y_test), dtype=np.int8)

        # collect predictions for each example
        y_predicted = scikit_dt.predict(X_test)

        scikit_result.nodes = scikit_dt.tree_.node_count
        scikit_result.time = end - start
        scikit_result.precision = precision_score(y_test, y_predicted)
        scikit_result.recall = recall_score(y_test, y_predicted)
        scikit_result.avg_precision = average_precision_score(y_test, y_predicted)
        scikit_result.f1 = f1_score(y_test, y_predicted)
        scikit_result.accuracy = accuracy_score(y_test, y_predicted)
        scikit_result.matthews = matthews_corrcoef(y_test, y_predicted)

        scikit_results_list.append(scikit_result)

dt_test_res = get_mean_scores(dt_results_list)
scikit_test_res = get_mean_scores(scikit_results_list)

print('                      {}     {}'.format('dt', 'sklearn dt'))
print('         Nodes:     {}           {}'.format(dt_test_res.nodes, scikit_test_res.nodes))
print('          Time:     {0:.2f}           {1:.2f}'.format(dt_test_res.time, scikit_test_res.time))
print('     Precision:     {0:.2f}           {1:.2f}'.format(dt_test_res.precision, scikit_test_res.precision))
print('        Recall:     {0:.2f}           {1:.2f}'.format(dt_test_res.recall, scikit_test_res.recall))
print('Avg. Precision:     {0:.2f}           {1:.2f}'.format(dt_test_res.avg_precision, scikit_test_res.avg_precision))
print('            F1:     {0:.2f}           {1:.2f}'.format(dt_test_res.f1, scikit_test_res.f1))
print('      Accuracy:     {0:.2f}           {1:.2f}'.format(dt_test_res.accuracy, scikit_test_res.accuracy))
print('           MCC:     {0:.2f}           {1:.2f}'.format(dt_test_res.matthews, scikit_test_res.matthews))
