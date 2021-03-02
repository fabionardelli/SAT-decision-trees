import numpy as np
from decisiontree import DecisionTree

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

dt = DecisionTree()
example = [0, 0, 0, 1]
dt.fit(data, 5)
predicted_class = dt.predict(example)
print(predicted_class)

print(dt.nodes)
print(dt.tree_structure)

dt.draw_tree('tree.png')
