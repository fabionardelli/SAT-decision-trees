from core import get_solutions
from dataclasses import dataclass
import pydot


@dataclass
class Node:
    id: int
    x: int = None  # input feature assigned to the node
    leaf: bool = False
    y: int = 0  # only for a leaf node


class DecisionTree:
    # Class used to implement a decision tree classifier.
    # A decision tree is made up by a full binary tree,
    # a tree in which each node has exactly 0 or 2 children.
    # For example, a 5 node decision tree could be like this:
    #     1
    #    / \
    #   2   3
    #  / \
    # 4   5
    # Here, node 1 is the root, node 2 is an intermediate node
    # and nodes 3, 4 and 5 are leaves.

    def __init__(self):
        # Dictionary used to store the nodes of the tree.
        # The keys are the nodes' ids and the values are
        # Node objects, representing the nodes' data.
        # For example, for the 5 node tree represented above,
        # 'nodes' would be like:
        # {1: Node(id=1, x=2, leaf=False, y=0), 2: Node(id=2, x=1, leaf=False, y=0),
        #  3: Node(id=3, x=None, leaf=True, y=0), 4: Node(id=4, x=None, leaf=True, y=1),
        #  5: Node(id=5, x=None, leaf=True, y=0)}
        self.nodes = {}

        # Dictionary used to represent the tree structure.
        # The keys are the nodes' ids and the values are lists
        # of left and right children's ids, if the key node is
        # not a leaf, or None otherwise.
        # For example, for the 5 node tree represented above,
        # the tree_structure would be like:
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

        visited = set()  # already visited nodes

        # Scan each node in the tree
        for node in self.tree_structure:
            # If the node is not a leaf and has not been visited yet...
            if self.tree_structure[node] is not None and node not in visited:
                # add it to the pydot graph...
                g.add_node(pydot.Node('%s' % node, shape='circle'))
                # and to the visited set
                visited.add(node)

                # get its left and right children...
                left_child = self.tree_structure[node][0]
                right_child = self.tree_structure[node][1]

                # and add them to the pydot graph
                g.add_node(pydot.Node('%s' % left_child, shape='circle'))
                g.add_node(pydot.Node('%s' % right_child, shape='circle'))

                # Then add edges connecting the parent node to its children
                g.add_edge(pydot.Edge(node, left_child, color='black'))
                g.add_edge(pydot.Edge(node, right_child, color='black'))

        # Save the pydot graph as a png image
        g.write_png(file_name)
