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
        # Dictionary used to store the nodes of the tree.
        # The keys are the nodes' ids and the values are
        # Node objects, representing the nodes' data.
        self.nodes = {}

        # Dictionary used to represent the tree structure.
        # The keys are the nodes' ids and the values are lists
        # of left and right children's ids, if the key node is
        # not a leaf, or None otherwise.
        # For example, for a 5 node tree in which the 1st node is
        # parent of the 2nd and 3rd node, the 2nd node is
        # parent of the 4th and the 5th node, and nodes 3, 4 and 5
        # are leaves, the tree_structure would be like:
        # {1: [2, 3], 2: [4, 5], 3: None, 4: None, 5: None}
        self.tree_structure = {}

    def fit(self, training_data, target_nodes):
        """ Trains the model given a training set and a target number of nodes."""

        # solve the CSP

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
                self.tree_structure[k] = []
            else:
                self.nodes[k].leaf = True
                self.tree_structure[k] = None

        l_var = solution['l']
        for k, v in l_var.items():
            self.tree_structure[k].append(v)

        r_var = solution['r']
        for k, v in r_var.items():
            self.tree_structure[k].append(v)

        a_var = solution['a']
        for k, v in a_var.items():
            self.nodes[v].x = k

        c_var = solution['c']
        for k, v in c_var.items():
            self.nodes[k].y = v

    def predict(self, item):
        """ Predicts the class of the item passed as argument."""
        
        # create a dictionary of pairs (feature_number, feature_value)
        item_data = {i: item[i - 1] for i in range(1, len(item) + 1)}

        current_node = self.nodes[1]  # get the tree root

        while not current_node.leaf:
            if current_node.x in item_data:
                if item_data[current_node.x] == 0:
                    # next node is left child
                    next_node = self.nodes[self.tree_structure[current_node.id][0]]
                else:
                    # next node is right child
                    next_node = self.nodes[self.tree_structure[current_node.id][1]]

                current_node = next_node

        return current_node.y

    def draw_tree(self, file_name):
        """
        Draws the tree as .png image, using pydot.
        :return:
        """
        g = pydot.Dot('tree', graph_type='digraph')

        # Scan each node in the tree
        for node in self.tree_structure:
            # If the node is not a leaf...
            if self.tree_structure[node] is not None:
                # Get its left and right children...
                left_child = self.tree_structure[node][0]
                right_child = self.tree_structure[node][1]

                # And add them to the pydot graph
                g.add_node(pydot.Node('%s' % left_child, shape='circle'))
                g.add_node(pydot.Node('%s' % right_child, shape='circle'))

                # Then add edges connecting the parent node to its children
                g.add_edge(pydot.Edge(node, left_child, color='black'))
                g.add_edge(pydot.Edge(node, right_child, color='black'))

        # Save the pydot graph as a png image
        g.write_png(file_name)