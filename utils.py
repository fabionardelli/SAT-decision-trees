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
