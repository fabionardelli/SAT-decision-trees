import numpy as np
from ortools.sat.python import cp_model
from math import floor
import traceback

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

# boolean mask to select only the rows where the target feature equals 1
mask = data[:, -1] == 1
pos_x = data[mask, :-1]  # input features for the positive examples

# boolean mask to select only the rows where the target feature equals 0
mask = data[:, -1] == 0
neg_x = data[mask, :-1]  # input features for the negative examples

n = 5  # number of nodes
k = len(pos_x[0])  # number of input features


def getLR(i):
    return tuple([_ for _ in range(i + 1, min(2 * i, n - 1) + 1) if _ % 2 == 0])


def getRR(i):
    return tuple([_ for _ in range(i + 2, min(2 * i + 1, n) + 1) if _ % 2 != 0])


# Create the model.
model = cp_model.CpModel()

# Create the variables.

# Create a dictionary to store all the variables
var = {}

# Create a variable 'v' for each node in the tree
# It is True iff node i is a leaf node
for i in range(1, n + 1):
    var['v%i' % i] = model.NewBoolVar('v%i' % i)

# Create a variable 'l' for each possible left child of current node
# It is True iff node i has node j as left child
for i in range(1, n + 1):
    for j in getLR(i):
        var['l%i,%i' % (i, j)] = model.NewBoolVar('l%i,%i' % (i, j))

# Create a variable 'r' for each possible right child of current node
# It is True iff node i has node j as right child
for i in range(1, n + 1):
    for j in getRR(i):
        var['r%i,%i' % (i, j)] = model.NewBoolVar('r%i,%i' % (i, j))

# Create a variable 'p' to tell the parent of current node
# It is True iff the parent of node j is node i
for i in range(1, n):
    for j in range(i + 1, min(2 * i + 1, n) + 1):
        var['p%i,%i' % (j, i)] = model.NewBoolVar('p%i,%i' % (j, i))

# Create a variable 'a' for each couple (node, feature)
# It is True iff feature r is assigned to node j
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['a%i,%i' % (i, j)] = model.NewBoolVar('a%i,%i' % (i, j))

# Create a variable 'u' for each couple (node, feature)
# It is True iff feature r is being discriminated against by node j
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['u%i,%i' % (i, j)] = model.NewBoolVar('u%i,%i' % (i, j))

# Create a variable 'd0' for each couple (node, feature)
# It is True iff feature r is discriminated for value 0 by node j
# or by one of its ancestors
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['d0%i,%i' % (i, j)] = model.NewBoolVar('d0%i,%i' % (i, j))

# Create a variable 'd1' for each couple (node, feature)
# It is True iff feature r is discriminated for value 1 by node j
# or by one of its ancestors
for i in range(1, k + 1):
    for j in range(1, n + 1):
        var['d1%i,%i' % (i, j)] = model.NewBoolVar('d1%i,%i' % (i, j))

# Create a variable 'c' for each node.
# It is True iff class of leaf node j is 1
for j in range(1, n + 1):
    var['c%i' % j] = model.NewBoolVar('c%i' % j)


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
#'''
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
# li,j -> lh,(j-2), and ri,j -> rh,(j-2), h < i
for i in range(n - 2, 0, -1):
    for j in reversed(getLR(i)):
        if 'l%i,%i' % (i, j) in var:
            s = 0
            for h in range(i - 1, 0, -1):
                if 'l%i,%i' % (h, j - 2) in var:
                    s += var['l%i,%i' % (h, j - 2)]
            if s > 0:
                model.Add(s >= 1).OnlyEnforceIf(var['l%i,%i' % (i, j)])
    for j in reversed(getRR(i)):
        if 'r%i,%i' % (i, j) in var:
            s = 0
            for h in range(i - 1, 0, -1):
                if 'r%i,%i' % (h, j - 2) in var:
                    s += var['r%i,%i' % (h, j - 2)]
            if s > 0:
                model.Add(s >= 1).OnlyEnforceIf(var['r%i,%i' % (i, j)])
#'''
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
def var_and(a, b):
    return a.GetVarValueMap()[1] == 1 and b.GetVarValueMap()[1] == 1


# Constraint 7: to discriminate a feature for value 0 at node j = 2,...,n
for r in range(1, k + 1):
    model.Add(var['d0%i,%i' % (r, 1)] == 0)

    for j in range(2, n + 1):
        or_sum = 0
        for i in range(floor(j / 2), j):
            if i >= 1 and j in getLR(i) or j in getRR(i):
                #print(var['p%i,%i' % (j, i)], var['d0%i,%i' % (r, i)])
                if var_and(var['p%i,%i' % (j, i)], var['d0%i,%i' % (r, i)]):
                    or_sum += 1
                else:
                    if 'r%i,%i' % (i, j) in var:
                        #print(var['a%i,%i' % (r, i)], var['r%i,%i' % (i, j)])
                        if var_and(var['a%i,%i' % (r, i)], var['r%i,%i' % (i, j)]):
                            or_sum += 1
        model.Add(or_sum >= 1).OnlyEnforceIf(var['d0%i,%i' % (r, j)])
        # model.Add(var['d0%i,%i' % (r, j)] == 1).OnlyEnforceIf(or_sum >= 1)
        model.AddImplication(or_sum >= 1, var['d0%i,%i' % (r, j)])

# Constraint 8: to discriminate a feature for value 1 at node j = 2,...,n
for r in range(1, k + 1):
    model.Add(var['d1%i,%i' % (r, 1)] == 0)

    for j in range(2, n + 1):
        or_sum = 0
        for i in range(floor(j / 2), j):
            if i >= 1 and j in getLR(i) or j in getRR(i):
                if var_and(var['p%i,%i' % (j, i)], var['d1%i,%i' % (r, i)]):
                    or_sum += 1
                else:
                    if 'l%i,%i' % (i, j) in var:
                        if var_and(var['a%i,%i' % (r, i)], var['l%i,%i' % (i, j)]):
                            or_sum += 1
        model.Add(or_sum >= 1).OnlyEnforceIf(var['d1%i,%i' % (r, j)])
        # model.Add(var['d0%i,%i' % (r, j)] == 1).OnlyEnforceIf(or_sum >= 1)
        model.AddImplication(or_sum >= 1, var['d1%i,%i' % (r, j)])
'''

# Constraint 7: to discriminate a feature for value 0 at node j = 2,...,n
for r in range(1, k + 1):
    model.Add(var['d0%i,%i' % (r, 1)] == 0)

    for j in range(2, n + 1):
        or_list = []
        for i in range(floor(j / 2), j):
            if i >= 1 and (j in getLR(i) or j in getRR(i)) and 'r%i,%i' % (i, j) in var:
                var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)] = model.NewBoolVar('__p%i,%i_AND_d0%i,%i' % (j, i, r, i))
                var['__a%i,%i_AND_r%i,%i' % (r, i, i, j)] = model.NewBoolVar('__a%i,%i_AND_r%i,%i' % (r, i, i, j))

                model.Add(var['p%i,%i' % (j, i)] + var['d0%i,%i' % (r, i)] == 2).OnlyEnforceIf(
                    var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)])
                model.Add(var['a%i,%i' % (r, i)] + var['r%i,%i' % (i, j)] == 2).OnlyEnforceIf(
                    var['__a%i,%i_AND_r%i,%i' % (r, i, i, j)])

                var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)] = model.NewBoolVar(
                    '__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j))

                model.Add(var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)] + var[
                    '__a%i,%i_AND_r%i,%i' % (r, i, i, j)] >= 1).OnlyEnforceIf(
                    var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)])

                or_list.append(var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)])

        model.AddBoolOr(or_list).OnlyEnforceIf(var['d0%i,%i' % (r, j)])
        model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['d0%i,%i' % (r, j)])

# Constraint 8: to discriminate a feature for value 1 at node j = 2,...,n
for r in range(1, k + 1):
    model.Add(var['d1%i,%i' % (r, 1)] == 0)

    for j in range(2, n + 1):
        or_list = []
        for i in range(floor(j / 2), j):
            if i >= 1 and (j in getLR(i) or j in getRR(i)) and 'l%i,%i' % (i, j) in var:
                var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)] = model.NewBoolVar('__p%i,%i_AND_d1%i,%i' % (j, i, r, i))
                var['__a%i,%i_AND_l%i,%i' % (r, i, i, j)] = model.NewBoolVar('__a%i,%i_AND_l%i,%i' % (r, i, i, j))

                model.Add(var['p%i,%i' % (j, i)] + var['d1%i,%i' % (r, i)] == 2).OnlyEnforceIf(
                    var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)])
                model.Add(var['a%i,%i' % (r, i)] + var['l%i,%i' % (i, j)] == 2).OnlyEnforceIf(
                    var['__a%i,%i_AND_l%i,%i' % (r, i, i, j)])

                var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)] = model.NewBoolVar(
                    '__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j))

                model.Add(var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)] + var[
                    '__a%i,%i_AND_l%i,%i' % (r, i, i, j)] >= 1).OnlyEnforceIf(
                    var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)])

                or_list.append(var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)])

        model.AddBoolOr(or_list).OnlyEnforceIf(var['d1%i,%i' % (r, j)])
        model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['d1%i,%i' % (r, j)])

'''
# Constraint 9: using a feature r at node j, r = 1,...,k, j = 1,...,n
for r in range(1, k + 1):
    for j in range(1, n + 1):
        or_sum = 0
        #or_exp = 0
        for i in range(floor(j / 2), j):
            if i >= 1 and j in getLR(i) or j in getRR(i):
                # ur,i ^ pj,i -> -ar,j
                model.Add(var['a%i,%i' % (r, j)] == 0).OnlyEnforceIf([var['u%i,%i' % (r, i)], var['p%i,%i' % (j, i)]])
                if var_and(var['u%i,%i' % (r, i)], var['p%i,%i' % (j, i)]):
                    or_sum += 1
                    #or_exp += or_sum
        if var['a%i,%i' % (r, j)] == 1:
            #or_exp += 1
            or_sum += 1
        model.Add(or_sum >= 1).OnlyEnforceIf(var['u%i,%i' % (r, j)])
        model.AddImplication(or_sum >= 1, var['u%i,%i' % (r, j)])
'''

# Constraint 9: using a feature r at node j, r = 1,...,k, j = 1,...,n
for r in range(1, k + 1):
    for j in range(1, n + 1):
        or_list = []
        for i in range(floor(j / 2), j):
            if i >= 1: #and j in getLR(i) or j in getRR(i):
                # ur,i ^ pj,i -> -ar,j
                var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)] = model.NewBoolVar('__u%i,%i_AND_p%i,%i' % (r, i, j, i))
                model.Add(var['u%i,%i' % (r, i)] + var['p%i,%i' % (j, i)] == 2).OnlyEnforceIf(
                    var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)])
                model.AddImplication(var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)], var['a%i,%i' % (r, j)].Not())

                or_list.append(var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)])

        # ar,j
        or_list.append(var['a%i,%i' % (r, j)])

        model.AddBoolOr(or_list).OnlyEnforceIf(var['u%i,%i' % (r, j)])
        model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['u%i,%i' % (r, j)])

        # avoid duplicates
        model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) == 0, var['u%i,%i' % (r, j)].Not())

# Constraint 10: for a non-leaf node j, exactly one feature is used
for j in range(1, n + 1):
    s = 0
    for r in range(1, k + 1):
        s += var['a%i,%i' % (r, j)]
    model.Add(s == 1).OnlyEnforceIf(var['v%i' % j].Not())

# Constraint 11: for a leaf node j, no feature is used
for j in range(1, n + 1):
    s = 0
    for r in range(1, k + 1):
        s += var['a%i,%i' % (r, j)]
    model.Add(s == 0).OnlyEnforceIf(var['v%i' % j])

# Constraint 12: any positive example must be discriminated if the leaf
# node is associated with the negative class
for example in pos_x:
    for j in range(1, n + 1):
        or_list = []
        for r in range(1, k + 1):
            if example[r - 1] == 0:
                or_list.append(var['d0%i,%i' % (r, j)])
            else:
                or_list.append(var['d1%i,%i' % (r, j)])
        model.AddBoolOr(or_list).OnlyEnforceIf([var['v%i' % j], var['c%i' % j].Not()])

# Constraint 13: any negative example must be discriminated if the leaf
# node is associated with the positive class
for example in neg_x:
    for j in range(1, n + 1):
        or_list = []
        for r in range(1, k + 1):
            if example[r - 1] == 0:
                or_list.append(var['d0%i,%i' % (r, j)])
            else:
                or_list.append(var['d1%i,%i' % (r, j)])
        model.AddBoolOr(or_list).OnlyEnforceIf([var['v%i' % j], var['c%i' % j]])

# Constraint 13-bis: only a leaf node can be associated to a class
for i in range(1, n + 1):
    model.AddImplication(var['c%i' % i], var['v%i' % i])

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
        a_var = {}
        u_var = {}
        d0_var = {}
        d1_var = {}
        c_var = {}

        for v in self.__variables:
            try:
                if str(v).startswith('v'):
                    v_var[int(str(v)[1:])] = self.Value(v)
                elif str(v).startswith('l') and self.Value(v) == 1:
                    parent = int(str(v)[1:].partition(',')[0])
                    child = int(str(v)[1:].partition(',')[2])
                    l_var[parent] = child
                elif str(v).startswith('r') and self.Value(v) == 1:
                    parent = int(str(v)[1:].partition(',')[0])
                    child = int(str(v)[1:].partition(',')[2])
                    r_var[parent] = child
                elif str(v).startswith('p') and self.Value(v) == 1:
                    child = int(str(v)[1:].partition(',')[0])
                    parent = int(str(v)[1:].partition(',')[2])
                    p_var[child] = parent
                elif str(v).startswith('a') and self.Value(v) == 1:
                    feature = int(str(v)[1:].partition(',')[0])
                    node = int(str(v)[1:].partition(',')[2])
                    a_var[feature] = node
                    '''
                elif str(v).startswith('u') and self.Value(v) == 1:
                    feature = int(str(v)[1:].partition(',')[0])
                    node = int(str(v)[1:].partition(',')[2])
                    u_var[feature] = node
                elif str(v).startswith('d0') and self.Value(v) == 1:
                    feature = int(str(v)[2:].partition(',')[0])
                    node = int(str(v)[2:].partition(',')[2])
                    d0_var[feature] = node
                elif str(v).startswith('d1') and self.Value(v) == 1:
                    feature = int(str(v)[2:].partition(',')[0])
                    node = int(str(v)[2:].partition(',')[2])
                    d1_var[feature] = node
                    '''
                elif str(v).startswith('c') and self.Value(v) == 1:
                    c_var[int(str(v)[1:])] = self.Value(v)
            except Exception as e:
                traceback.print_exc()
                raise e

        solution = {'v': v_var, 'l': l_var, 'r': r_var, 'a': a_var, 'c': c_var}

        if solution not in self.solution_list:
            self.solution_list.append(solution)


def get_solutions():
    # Create a solver and solve the model.
    solver = cp_model.CpSolver()
    solution_collector = VarArraySolutionCollector(var.values())
    solver.SearchForAllSolutions(model, solution_collector)
    #solver.SolveWithSolutionCallback(model, solution_collector)

    # return tuple(__ for __ in solution_collector.solution_list)
    return tuple(solution_collector.solution_list)


sol = get_solutions()
for item in sol:
    print(item)
print(len(sol))
