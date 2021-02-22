from core import get_solutions
import pydot


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
