import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from math import floor

'''
df = pd.read_csv('data.csv')
features = df.iloc[:, :-1]  # get input features from dataset
target = df.iloc[:, -1]  # get target feature from dataset
'''

n = 5  # number of nodes
k = 5  # number of input features


def getLR(i):
    return tuple([_ for _ in range(i + 1, min(2 * i, n - 1) + 1) if _ % 2 == 0])


def getRR(i):
    return tuple([_ for _ in range(i + 2, min(2 * i + 1, n) + 1) if _ % 2 != 0])


# Creates the model.
model = cp_model.CpModel()

# Creates the variables.

# Create a dictionary to store all the variables
var = {}

# Creates a variable 'v' for each node in the tree
# It is True iff node i is a leaf node
for i in range(1, n + 1):
    var['v%i' % i] = model.NewBoolVar('v%i' % i)

# Creates a variable 'l' for each possible left child of current node
# It is True iff node i has node j as left child
for i in range(1, n + 1):
    for j in getLR(i):
        var['l%i,%i' % (i, j)] = model.NewBoolVar('l%i,%i' % (i, j))

# Creates a variable 'r' for each possible right child of current node
# It is True iff node i has node j as right child
for i in range(1, n + 1):
    for j in getRR(i):
        var['r%i,%i' % (i, j)] = model.NewBoolVar('r%i,%i' % (i, j))

# Creates a variable 'p' to tell the parent of current node
# It is True iff the parent of node j is node i
for i in range(1, n):
    for j in range(i + 1, min(2 * i + 1, n) + 1):
        var['p%i,%i' % (j, i)] = model.NewBoolVar('p%i,%i' % (j, i))
'''
# Creates a variable 'a' for each couple (node, feature)
# It is True iff feature r is assigned to node j
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['a%i,%i' % (i, j)] = model.NewBoolVar('a%i,%i' % (i, j))

# Creates a variable 'u' for each couple (node, feature)
# It is True iff feature r is being discriminated against by node j
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['u%i,%i' % (i, j)] = model.NewBoolVar('u%i,%i' % (i, j))

# Creates a variable 'd0' for each couple (node, feature)
# It is True iff feature r is discriminated for value 0 by node j
# or by one of its ancestors
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['d0%i,%i' % (i, j)] = model.NewBoolVar('d0%i,%i' % (i, j))

# Creates a variable 'd1' for each couple (node, feature)
# It is True iff feature r is discriminated for value 1 by node j
# or by one of its ancestors
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['d1%i,%i' % (i, j)] = model.NewBoolVar('d1%i,%i' % (i, j))

# Creates a variable 'c' for each node.
# It is True iff class of leaf node j is 1
for j in range(1, n + 1):
    var['c%i' % i] = model.NewBoolVar('c%i' % j)
'''
# Constraints.

# TREE BUILDING CONSTRAINTS
# These constraints allow to represent a full binary tree, that is a tree
# in which each node has exactly 0 or 2 children. Since the tree cannot be
# made up by the root alone, the number of nodes n must be an odd number >= 3.
# Given n, the constraints permit to find all the possible full binary trees
# which can be built with n nodes. The number of legit trees with n nodes
# is given by the Catalan numbers sequence.

# Constraint 1: the root node is not a leaf
model.Add(var['v1'] == 0)

# Constraint 2: if a node is a leaf node, then it has no children
for i in range(1, n + 1):
    for j in getLR(i):
        model.AddImplication(var['v%i' % i], var['l%i,%i' % (i, j)].Not())

# Constraint 3: the left child and the right child of the i-th node
# are numbered consecutively
for i in range(1, n + 1):
    for j in getLR(i):
        model.AddImplication(var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j + 1)])
        model.AddImplication(var['r%i,%i' % (i, j + 1)], var['l%i,%i' % (i, j)])

# Constraint 4: a non-leaf node must have a child
for i in range(1, n + 1):
    s = 0
    for j in getLR(i):
        s += var['l%i,%i' % (i, j)]
    model.Add(s == 1).OnlyEnforceIf(var['v%i' % i].Not())

# Constraint 4bis: each left/right child must have exactly a parent
for j in range(2, n + 1):
    left_sum = 0
    right_sum = 0
    for i in range(1, n):
        if 'l%i,%i' % (i, j) in var:
            left_sum += var['l%i,%i' % (i, j)]
        if 'r%i,%i' % (i, j) in var:
            right_sum += var['r%i,%i' % (i, j)]
        if left_sum > 0:
            model.Add(left_sum <= 1)
        if right_sum > 0:
            model.Add(right_sum <= 1)

# Constraint 4ter: nodes on the same level must be labeled increasingly
# li,j -> lk,(j-2), and ri,j -> rk,(j-2), k < i
for i in range(n - 2, 0, -1):
    for j in reversed(getLR(i)):
        if 'l%i,%i' % (i, j) in var:
            s = 0
            for k in range(i - 1, 0, -1):
                if 'l%i,%i' % (k, j - 2) in var:
                    s += var['l%i,%i' % (k, j - 2)]
            if s > 0:
                model.Add(s >= 1).OnlyEnforceIf(var['l%i,%i' % (i, j)])
    for j in reversed(getRR(i)):
        if 'r%i,%i' % (i, j) in var:
            s = 0
            for k in range(i - 1, 0, -1):
                if 'r%i,%i' % (k, j - 2) in var:
                    s += var['r%i,%i' % (k, j - 2)]
            if s > 0:
                model.Add(s >= 1).OnlyEnforceIf(var['r%i,%i' % (i, j)])

# Constraint 5: if the i-th node is a parent then it must have a child
for i in range(1, n + 1):
    for j in getLR(i):
        model.AddImplication(var['p%i,%i' % (j, i)], var['l%i,%i' % (i, j)])
        model.AddImplication(var['l%i,%i' % (i, j)], var['p%i,%i' % (j, i)])
    for j in getRR(i):
        model.AddImplication(var['p%i,%i' % (j, i)], var['r%i,%i' % (i, j)])
        model.AddImplication(var['r%i,%i' % (i, j)], var['p%i,%i' % (j, i)])

# Constraint 6: all the nodes but the first must have a parent
for j in range(2, n + 1):
    s = []
    for i in range(floor(j / 2), min(j - 1, n) + 1):
        s.append(var['p%i,%i' % (j, i)])
    model.AddBoolXOr(s)

# LEARNING CONSTRAINTS
# These constraints allow to build a decision tree starting from a
# dataset of binary features.



'''
# To print all solutions
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count

solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinter(var.values())
status = solver.SearchForAllSolutions(model, solution_printer)

print('Status = %s' % solver.StatusName(status))
print('Number of solutions found: %i' % solution_printer.solution_count())
'''


# Stores all the solutions in a list
class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solution_list = []

    def on_solution_callback(self):
        # self.solution_list.append(['%s=%i' % (v, self.Value(v)) for v in self.__variables])
        # self.solution_list.append({'%s' % v: self.Value(v) for v in self.__variables})
        v_var = {}
        l_var = {}
        r_var = {}
        p_var = {}

        for v in self.__variables:
            if 'v' in str(v).lower():
                v_var['%i' % int(str(v)[1:])] = self.Value(v)
            elif 'l' in str(v).lower() and self.Value(v) == 1:
                parent = int(str(v)[1:].partition(',')[0])
                child = int(str(v)[1:].partition(',')[2])
                l_var[parent] = child
            elif 'r' in str(v).lower() and self.Value(v) == 1:
                parent = int(str(v)[1:].partition(',')[0])
                child = int(str(v)[1:].partition(',')[2])
                r_var[parent] = child
            elif 'p' in str(v).lower() and self.Value(v) == 1:
                child = int(str(v)[1:].partition(',')[0])
                parent = int(str(v)[1:].partition(',')[2])
                p_var[child] = parent

        solution = {'v': v_var, 'l': l_var, 'r': r_var, 'p': p_var}
        self.solution_list.append(solution)


def get_solutions():
    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    solution_collector = VarArraySolutionCollector(var.values())
    solver.SearchForAllSolutions(model, solution_collector)

    # return tuple(__ for __ in solution_collector.solution_list)
    return tuple(solution_collector.solution_list)


sol = get_solutions()
for s in sol:
    print(s)
