import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score, f1_score, \
    accuracy_score, matthews_corrcoef
from sklearn import tree
from decisiontree import DecisionTree
import time
from utils import ResultSet, get_mean_scores
import random


dt = DecisionTree()
scikit_dt = tree.DecisionTreeClassifier()

# read a dataset
data = pd.read_csv('datasets/binary/bin-car.csv', delimiter=',', header=None)
#data = pd.read_csv('datasets/binary/breast-cancer-un.csv', delimiter=',', header=None)
#data = pd.read_csv('datasets/binary/heart-cleveland-un.csv', delimiter=',', header=None)

X = data.iloc[:, :-1].to_numpy(dtype=np.int8)
y = data.iloc[:, -1].to_numpy(dtype=np.int8)

dt_results_list = []
scikit_results_list = []

random.seed(17)

# generate 40 random seeds for dataset sampling
seeds = random.sample(range(1, 32768), 40)

for s in seeds:
    # split dataset in training and test set

    # setting for "car" dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1708, random_state=s)

    # setting for "breast-cancer" dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=633, random_state=s)

    # setting for "heart cleveland" dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=281, random_state=s)
    
    start = time.time()
    n = dt.fit_optimal(X_train, y_train)
    # n = dt.fit(X_train, y_train, 5)
    end = time.time()

    # for test purposes, to check correctness of classifier
    # X_test = X_train
    # y_test = y_train

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
         Nodes:     11.3           9.2
          Time:     0.51           0.00
     Precision:     0.69           0.63
        Recall:     0.72           0.60
Avg. Precision:     0.60           0.51
            F1:     0.69           0.59
      Accuracy:     0.82           0.77
           MCC:     0.57           0.45


Comparison of execution times with different implementations
and no additional constraints
                      
yices    ortools  z3
2.13     4.89     11.56    

    
Execution time with additional constraints (yices implementation)
With 6.1
2.11  

With 6.2
1.04

With 6.1 and 6.2
0.99

With 13.1
2.14

With 6.1 and 13.1
1.96

With 6.2 and 13.1
0.98

With 6.1, 6.2 and 13.1
1.03

With original additional constraints
1.00
      
With 6.1 and original additional constraints
1.02
           
With 6.2 and original additional constraints
1.07
           
With 13.1 and original additional constraints
1.02

With 6.1, 13.1 and original additional constraints
1.20
           
with 6.2, 13.1 and original additional constraints
0.99
           
With all constraints
1.05


results with 40 run

"Car" dataset, 20 examples
                      dt     sklearn dt
         Nodes:     10.65          9.05
          Time:     0.54           0.00
     Precision:     0.68           0.61
        Recall:     0.72           0.60
Avg. Precision:     0.59           0.50
            F1:     0.69           0.58
      Accuracy:     0.81           0.75
           MCC:     0.57           0.43
         
           
"Breast cancer dataset", 50 examples
                      dt     sklearn dt
         Nodes:     8.95           8.7
          Time:     9.64           0.00
     Precision:     0.86           0.85
        Recall:     0.82           0.85
Avg. Precision:     0.76           0.78
            F1:     0.83           0.85
      Accuracy:     0.89           0.89
           MCC:     0.75           0.77
           
"Heart Cleveland" dataset, 10 examples
                     dt     sklearn dt
         Nodes:     4.4            4.4
          Time:     0.34           0.00
     Precision:     0.71           0.69
        Recall:     0.67           0.66
Avg. Precision:     0.66           0.64
            F1:     0.67           0.65
      Accuracy:     0.67           0.65
           MCC:     0.37           0.32
        
"Heart Cleveland" dataset, 15 examples
                      dt     sklearn dt
         Nodes:     6.3            6.4
          Time:     1.96           0.00
     Precision:     0.69           0.71
        Recall:     0.73           0.70
Avg. Precision:     0.65           0.66
            F1:     0.70           0.69
      Accuracy:     0.67           0.67
           MCC:     0.35           0.36
'''
