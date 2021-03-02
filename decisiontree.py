from core import get_solutions
from dataclasses import dataclass
import pydot

'''
class Node:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value


class BinTree:
    root = 0

    def __init__(self, nodes):
        self.num_nodes = len(nodes)
        self.nodes = nodes

    def get_root(self):
        return self.root

    def left_child(self, n):
        try:
            return self.nodes[2 * n + 1]
        except IndexError as e:
            print(e)

    def right_child(self, n):
        try:
            return self.nodes[2 * n + 2]
        except IndexError as e:
            print(e)
            
            


n1 = Node('a')
n2 = Node('b')
n3 = Node('c')

# t = BinTree([n1, n2, n3])
# print(t.left_child(t.get_root()))
solutions = get_solutions()
for i, s in enumerate(solutions):
    g = pydot.Dot('t%i' % i, graph_type='digraph')
    v_var = s['v']
    for k in v_var:
        g.add_node(pydot.Node('%s' % k, shape='circle'))

    l_var = s['l']
    for k, v in l_var.items():
        g.add_edge(pydot.Edge(k, v, color='black'))

    r_var = s['r']
    for k, v in r_var.items():
        g.add_edge(pydot.Edge(k, v, color='black'))

    g.write_png('t%i.png' % i)
'''


@dataclass
class Node:
    id: int
    x: int = None  # input feature assigned to the node
    leaf: bool = False
    y: int = 0  # only for a leaf node


class DecisionTree:
    def __init__(self):
        self.nodes = {}
        self.tree = {}

    def fit(self, training_data, target_nodes):

        # solve the CSP
        solutions = []

        while True:
            solutions = get_solutions(training_data, target_nodes)
            if len(solutions) > 0:
                break
            else:
                target_nodes += 2

        solution = solutions[0]  # choose the first solution found

        # build the decision tree

        v_var = solution['v']
        for k, v in v_var.items():
            self.nodes[k] = Node(k)
            if v == 0:
                self.tree[k] = []
            else:
                self.nodes[k].leaf = True
                self.tree[k] = None

        l_var = solution['l']
        for k, v in l_var.items():
            self.tree[k].append(v)

        r_var = solution['r']
        for k, v in r_var.items():
            self.tree[k].append(v)

        a_var = solution['a']
        for k, v in a_var.items():
            self.nodes[v].x = k

        c_var = solution['c']
        for k, v in c_var.items():
            self.nodes[k].y = v

    def predict(self, item):
        # create a dictionary of pairs (feature_number, feature_value)
        item_data = {i: item[i - 1] for i in range(1, len(item) + 1)}

        current_node = self.nodes[1]  # get the tree root

        while not current_node.leaf:
            if current_node.x in item_data:
                if item_data[current_node.x] == 0:
                    # next node is left child
                    next_node = self.nodes[self.tree[current_node.id][0]]
                else:
                    # next node is right child
                    next_node = self.nodes[self.tree[current_node.id][1]]

                current_node = next_node

        return current_node.y
