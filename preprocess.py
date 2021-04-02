import numpy as np
import pandas as pd
from utils import one_hot_encode


# preprocess 'car' dataset
# load 'car' dataset
data = pd.read_csv('datasets/car.data', delimiter=',', header=None)

# boolean class values
y_data = data[6]
y_data.replace(to_replace=dict(unacc=0, acc=1, good=1, vgood=1), inplace=True)
y = y_data.to_numpy(dtype=np.int8)

# boolean features' values
X_data = data.drop(6, axis=1)
X = one_hot_encode(X_data)

binary_data = np.append(X, y[:, None], axis=1)

np.savetxt('datasets/binary/bin-car.csv', binary_data, fmt='%d', delimiter=',')
