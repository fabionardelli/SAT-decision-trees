import numpy as np


def make_boolean_dataset(data):
    """
        Converts a dataset of discrete features into a dataset of boolean features
        using One-Hot Encoding. Returns the boolean dataset.
    """

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

    return boolean_data
