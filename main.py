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

Without additional constraints
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     2.42           0.00
     Precision:     0.66           0.63
        Recall:     0.71           0.71
Avg. Precision:     0.56           0.54
            F1:     0.66           0.65
      Accuracy:     0.80           0.78
           MCC:     0.54           0.51

With 4bis
                      dt     sklearn dt
         Nodes:     13.4           10.2
          Time:     2.19           0.00
     Precision:     0.61           0.61
        Recall:     0.68           0.67
Avg. Precision:     0.53           0.52
            F1:     0.63           0.63
      Accuracy:     0.77           0.77
           MCC:     0.48           0.47

With 4ter
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.11           0.00
     Precision:     0.64           0.62
        Recall:     0.73           0.66
Avg. Precision:     0.56           0.53
            F1:     0.67           0.62
      Accuracy:     0.79           0.77
           MCC:     0.53           0.48

With 4bis and 4ter
                      dt     sklearn dt
         Nodes:     13.4           10.2
          Time:     1.07           0.00
     Precision:     0.66           0.62
        Recall:     0.71           0.67
Avg. Precision:     0.56           0.53
            F1:     0.67           0.63
      Accuracy:     0.80           0.77
           MCC:     0.54           0.48

With 13bis
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     2.72           0.00
     Precision:     0.64           0.63
        Recall:     0.69           0.71
Avg. Precision:     0.54           0.55
            F1:     0.65           0.65
      Accuracy:     0.79           0.78
           MCC:     0.51           0.51

With constraints 4ter and 13bis
                      dt     sklearn dt
         Nodes:     13.4           10.2
          Time:     1.09           0.00
     Precision:     0.63           0.62
        Recall:     0.70           0.65
Avg. Precision:     0.54           0.53
            F1:     0.65           0.62
      Accuracy:     0.78           0.77
           MCC:     0.51           0.48

With 4bis, 4ter and 13bis
                      dt     sklearn dt
         Nodes:     13.4           10.2
          Time:     1.09           0.00
     Precision:     0.64           0.62
        Recall:     0.71           0.66
Avg. Precision:     0.55           0.53
            F1:     0.66           0.62
      Accuracy:     0.79           0.77
           MCC:     0.52           0.48

With original additional constraints
                     dt     sklearn dt
         Nodes:     13.4           10.2
          Time:     1.19           0.00
     Precision:     0.62           0.60
        Recall:     0.69           0.68
Avg. Precision:     0.53           0.52
            F1:     0.64           0.63
      Accuracy:     0.77           0.76
           MCC:     0.49           0.47
           
With 4ter and original additional constraints
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.28           0.00
     Precision:     0.66           0.62
        Recall:     0.71           0.67
Avg. Precision:     0.57           0.53
            F1:     0.67           0.63
      Accuracy:     0.81           0.77
           MCC:     0.55           0.48
           
With 13bis and original additional constraints
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.20           0.00
     Precision:     0.64           0.64
        Recall:     0.70           0.69
Avg. Precision:     0.54           0.54
            F1:     0.65           0.65
      Accuracy:     0.78           0.78
           MCC:     0.51           0.51
           
with 4ter, 13bis and original additional constraints
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.18           0.00
     Precision:     0.64           0.62
        Recall:     0.72           0.68
Avg. Precision:     0.56           0.53
            F1:     0.67           0.63
      Accuracy:     0.79           0.77
           MCC:     0.53           0.48
           
With all constraints
                      dt     sklearn dt
         Nodes:     13.4           10.1
          Time:     1.13           0.00
     Precision:     0.64           0.63
        Recall:     0.71           0.69
Avg. Precision:     0.55           0.54
            F1:     0.66           0.64
      Accuracy:     0.79           0.78
           MCC:     0.52           0.50
'''