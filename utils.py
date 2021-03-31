import numpy as np


def one_hot_encode(data):
    """
        Converts a dataset of discrete features into a dataset of binary features
        using One-Hot Encoding. Returns the boolean dataset.
    """

    feature_domains = []  # to store features domains
    bin_col_number = 0  # num. of columns of the new dataset

    new_col = 0  # col index of current feature value in 'bin_data'
    for col, feature in data.iteritems():
        current_feature_domain = {}

        # store each feature's value in a dictionary of
        # pairs (value, new_col)
        for value in feature.values:
            if value not in current_feature_domain:
                current_feature_domain[value] = new_col
                new_col += 1

        feature_domains.append(current_feature_domain)
        bin_col_number += len(current_feature_domain)

    # this array will contain the binary dataset
    bin_data = np.zeros(shape=(data.shape[0], bin_col_number), dtype=np.int8)

    # populate the binary dataset
    for col, feature in data.iteritems():
        for row, value in enumerate(feature):
            bin_data[row, feature_domains[col][value]] = 1

    return bin_data


class ResultSet:
    """ Class to gather test results"""

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
    """ Returns a ResultSet object with the mean values of the metrics.
        Takes a list of ResultSet objects in input.
    """

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
