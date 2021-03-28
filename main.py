import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score, \
    accuracy_score, matthews_corrcoef
from sklearn import tree
from decisiontree import DecisionTree
import time


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


def get_mean_scores(res_list):
    """ Returns a ResultSet object with the mean values of the metrics."""

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


dt = DecisionTree()
scikit_dt = tree.DecisionTreeClassifier()

# read a dataset
data = pd.read_csv('datasets/binary/bin-car.csv', delimiter=',', header=None)
#data = pd.read_csv('data.csv', delimiter=',', header=None, skiprows=1)
X = data.iloc[:, :-1].to_numpy(dtype=np.int8)
y = data.iloc[:, -1].to_numpy(dtype=np.int8)


dt_results_list = []
scikit_results_list = []

for s in range(1, 21):
    # split dataset in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1708, random_state=s)

    start = time.time()
    n = dt.fit_optimal(X_train, y_train)
    #n = dt.fit(X_train, y_train, 5)
    end = time.time()

    #X_test = X_train
    #y_test = y_train

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

'''
Results with 20 runs with a sample of 20 examples from the car dataset


Comparison of classifier performance with sklearn DecisionTreeClassifier
(all constraints active)
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.05           0.00
     Precision:     0.66           0.62
        Recall:     0.71           0.67
Avg. Precision:     0.57           0.53
            F1:     0.67           0.63
      Accuracy:     0.81           0.77
           MCC:     0.55           0.48


Comparison of execution times with different implementations
and no additional constraints
                      
yices    ortools  z3
2.42     4.95     11.72    
    
    
Execution time with additional constraints (yices implementation)
With 4bis
2.19  

With 4ter
1.11

With 4bis and 4ter
1.07

With 13bis
2.72

With 4ter and 13bis
1.09

With 4bis, 4ter and 13bis
1.09

With original additional constraints
1.19
           
With 4ter and original additional constraints
1.28
           
With 13bis and original additional constraints
1.20
           
with 4ter, 13bis and original additional constraints
1.18
           
With all constraints
1.05
'''