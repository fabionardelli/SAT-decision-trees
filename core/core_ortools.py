# This software implements a method for learning a decision tree as a SAT problem,
# as shown in the paper "Learning Optimal Decision Trees with SAT" by Nina Narodytska et al., 2018
# All credit for the encoding proposed here must be given to them.
# Author: Fabio Nardelli

import numpy as np
from ortools.sat.python import cp_model
from math import floor, ceil
import traceback

# Global variables to be used later on.
model = None  # to store a model for Google OR-Tools
var = {}  # to store the variables of the csp


def set_csp(pos_x, neg_x, n, k):
    """Creates the variables and the constraints for a given number of nodes."""

    # Create the model.
    global model
    model = cp_model.CpModel()

    # Create a dictionary to store all the variables
    global var
    var = {}

    def get_lr(i):
        """
        Given a node i, returns all its possible left children.
        LR(i) = even([i + 1, min(2i, n − 1)]), i = 1,...,n
        """
        return tuple([_ for _ in range(i + 1, min(2 * i, n - 1) + 1) if _ % 2 == 0])

    def get_rr(i):
        """
        Given a node i, returns all its possible right children.
        RR(i) = odd([i + 2, min(2i + 1, n )]), i=1,...,n
        """
        return tuple([_ for _ in range(i + 2, min(2 * i + 1, n) + 1) if _ % 2 != 0])

    # Create the variables.

    # Create a variable 'v' for each node in the tree
    # It is True iff node i is a leaf node
    for i in range(1, n + 1):
        var['v%i' % i] = model.NewBoolVar('v%i' % i)

    # Create a variable 'l' for each possible left child of current node
    # It is True iff node i has node j as left child
    for i in range(1, n + 1):
        for j in get_lr(i):
            var['l%i,%i' % (i, j)] = model.NewBoolVar('l%i,%i' % (i, j))

    # Create a variable 'r' for each possible right child of current node
    # It is True iff node i has node j as right child
    for i in range(1, n + 1):
        for j in get_rr(i):
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
    # made up by the root alone, the number of nodes 'n' must be an odd number >= 3.
    # Given n, the constraints permit to find all the possible full binary trees
    # which can be built with n nodes. The number of legit trees with n nodes
    # is given by the Catalan numbers sequence.

    # Constraint 1: the root node is not a leaf.
    # NOT v1
    model.Add(var['v1'] == 0)

    # Constraint 2: if a node is a leaf node, then it has no children.
    # vi -> NOT li,j, j in LR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            model.AddImplication(var['v%i' % i], var['l%i,%i' % (i, j)].Not())

    # Constraint 3: the left child and the right child of the i-th node
    # are numbered consecutively.
    # li,j <-> ri,j+1, j in LR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            model.AddImplication(var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j + 1)])
            model.AddImplication(var['r%i,%i' % (i, j + 1)], var['l%i,%i' % (i, j)])

    # Constraint 4: a non-leaf node must have a child.
    # NOT vi -> (SUM for j in LR(i) of li,j = 1)
    for i in range(1, n + 1):
        s = 0
        for j in get_lr(i):
            s += var['l%i,%i' % (i, j)]
        model.Add(s == 1).OnlyEnforceIf(var['v%i' % i].Not())

    # Constraint 5: if the i-th node is a parent then it must have a child
    # pj,i <-> li,j, j in LR(i)
    # pj,i <-> ri,j, j in RR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            model.AddImplication(var['p%i,%i' % (j, i)], var['l%i,%i' % (i, j)])
            model.AddImplication(var['l%i,%i' % (i, j)], var['p%i,%i' % (j, i)])
        for j in get_rr(i):
            model.AddImplication(var['p%i,%i' % (j, i)], var['r%i,%i' % (i, j)])
            model.AddImplication(var['r%i,%i' % (i, j)], var['p%i,%i' % (j, i)])

    # Constraint 6: all the nodes but the first must have a parent.
    # (SUM for i=floor(j/2), min(j-1, N) of pj,i = 1), j =2,...,n
    for j in range(2, n + 1):
        s = []
        for i in range(floor(j / 2), min(j - 1, n) + 1):
            s.append(var['p%i,%i' % (j, i)])
        model.Add(sum(s) == 1)

    # '''
    # Constraint 6.1: each left/right child must have exactly a parent
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
    # '''
    # '''
    # Constraint 6.2: nodes on the same level must be labeled increasingly
    # li,j -> lh,(j-2), and ri,j -> rh,(j-2), h < i
    for i in range(n - 2, 0, -1):
        for j in reversed(get_lr(i)):
            if 'l%i,%i' % (i, j) in var:
                s = 0
                for h in range(i - 1, 0, -1):
                    if 'l%i,%i' % (h, j - 2) in var:
                        s += var['l%i,%i' % (h, j - 2)]
                if s > 0:
                    model.Add(s >= 1).OnlyEnforceIf(var['l%i,%i' % (i, j)])
        for j in reversed(get_rr(i)):
            if 'r%i,%i' % (i, j) in var:
                s = 0
                for h in range(i - 1, 0, -1):
                    if 'r%i,%i' % (h, j - 2) in var:
                        s += var['r%i,%i' % (h, j - 2)]
                if s > 0:
                    model.Add(s >= 1).OnlyEnforceIf(var['r%i,%i' % (i, j)])
    # '''

    # LEARNING CONSTRAINTS
    # These constraints allow to learn a decision tree starting from a
    # dataset of binary features. The resulting tree is represented
    # as a total assignment to the variables. The values of these variables
    # must be used to build a tree and implement the classifier.

    # Constraint 7: to discriminate a feature for value 0 at node j = 2,...,n
    # d0r,j <-> (OR for i=floor(j/2), j-1 of ((pj,i AND d0r,i) OR (ar,i AND ri,j)))
    # d0r,1 = 0
    for r in range(1, k + 1):
        model.Add(var['d0%i,%i' % (r, 1)] == 0)  # d0r,1 = 0

        for j in range(2, n + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                # Here we use intermediate variables to represent complex constraints.
                # Intermediate variables start with __.
                if i >= 1 and (j in get_lr(i) or j in get_rr(i)) and 'r%i,%i' % (i, j) in var:
                    var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)] = model.NewBoolVar('__p%i,%i_AND_d0%i,%i' % (j, i, r, i))
                    var['__a%i,%i_AND_r%i,%i' % (r, i, i, j)] = model.NewBoolVar('__a%i,%i_AND_r%i,%i' % (r, i, i, j))

                    # __p%i,%i_AND_d0%i,%i -> p%i,%i AND d0%i,%i
                    model.Add(var['p%i,%i' % (j, i)] + var['d0%i,%i' % (r, i)] == 2).OnlyEnforceIf(
                        var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)])

                    # __a%i,%i_AND_r%i,%i -> a%i,%i AND r%i,%i
                    model.Add(var['a%i,%i' % (r, i)] + var['r%i,%i' % (i, j)] == 2).OnlyEnforceIf(
                        var['__a%i,%i_AND_r%i,%i' % (r, i, i, j)])

                    var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)] = model.NewBoolVar(
                        '__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j))

                    # __p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i -> __p%i,%i_AND_d0%i,%i AND __a%i,%i_AND_r%i,%i
                    model.Add(var['__p%i,%i_AND_d0%i,%i' % (j, i, r, i)] + var[
                        '__a%i,%i_AND_r%i,%i' % (r, i, i, j)] >= 1).OnlyEnforceIf(
                        var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)])

                    or_list.append(var['__p%i,%i_AND_d0%i,%i_OR_a%i,%i_AND_r%i,%i' % (j, i, r, i, r, i, i, j)])

            # d0r,j -> (OR for i=floor(j/2), j-1 of ((pj,i AND d0r,i) OR (ar,i AND ri,j)))
            model.AddBoolOr(or_list).OnlyEnforceIf(var['d0%i,%i' % (r, j)])
            # d0r,j <- (OR for i=floor(j/2), j-1 of ((pj,i AND d0r,i) OR (ar,i AND ri,j)))
            model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['d0%i,%i' % (r, j)])

    # Constraint 8: to discriminate a feature for value 1 at node j = 2,...,n
    # d1r,j <-> (OR for i=floor(j/2), j-1 of ((pj,i AND d1r,i) OR (ar,i AND li,j)))
    # d1r,1 = 0
    for r in range(1, k + 1):
        model.Add(var['d1%i,%i' % (r, 1)] == 0)  # d1r,1 = 0

        for j in range(2, n + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                # Here we use intermediate variables to represent complex constraints.
                # Intermediate variables start with __.
                if i >= 1 and (j in get_lr(i) or j in get_rr(i)) and 'l%i,%i' % (i, j) in var:
                    var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)] = model.NewBoolVar('__p%i,%i_AND_d1%i,%i' % (j, i, r, i))
                    var['__a%i,%i_AND_l%i,%i' % (r, i, i, j)] = model.NewBoolVar('__a%i,%i_AND_l%i,%i' % (r, i, i, j))

                    # __p%i,%i_AND_d1%i,%i -> p%i,%i AND d1%i,%i
                    model.Add(var['p%i,%i' % (j, i)] + var['d1%i,%i' % (r, i)] == 2).OnlyEnforceIf(
                        var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)])

                    # __a%i,%i_AND_l%i,%i -> a%i,%i AND l%i,%i
                    model.Add(var['a%i,%i' % (r, i)] + var['l%i,%i' % (i, j)] == 2).OnlyEnforceIf(
                        var['__a%i,%i_AND_l%i,%i' % (r, i, i, j)])

                    var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)] = model.NewBoolVar(
                        '__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j))

                    # __p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i -> __p%i,%i_AND_d1%i,%i AND __a%i,%i_AND_l%i,%i
                    model.Add(var['__p%i,%i_AND_d1%i,%i' % (j, i, r, i)] + var[
                        '__a%i,%i_AND_l%i,%i' % (r, i, i, j)] >= 1).OnlyEnforceIf(
                        var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)])

                    or_list.append(var['__p%i,%i_AND_d1%i,%i_OR_a%i,%i_AND_l%i,%i' % (j, i, r, i, r, i, i, j)])

            # d1r,j -> (OR for i=floor(j/2), j-1 of ((pj,i AND d1r,i) OR (ar,i AND li,j)))
            model.AddBoolOr(or_list).OnlyEnforceIf(var['d1%i,%i' % (r, j)])
            # d1r,j <- (OR for i=floor(j/2), j-1 of ((pj,i AND d1r,i) OR (ar,i AND li,j)))
            model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['d1%i,%i' % (r, j)])

    # Constraint 9: using a feature r at node j, r = 1,...,k, j = 1,...,n
    # AND for i = floor(j/2), j-1 of (ur,i ^ pj,i -> -ar,j)
    # ur,j <-> (ar,j OR (OR for i = floor(j/2), j-1 of (ur,j AND pj,i)))
    for r in range(1, k + 1):
        for j in range(1, n + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                # Here we use intermediate variables to represent complex constraints.
                # Intermediate variables start with __.
                if i >= 1:  # and j in getLR(i) or j in getRR(i):
                    # ur,i ^ pj,i -> -ar,j
                    var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)] = model.NewBoolVar('__u%i,%i_AND_p%i,%i' % (r, i, j, i))

                    # __u%i,%i_AND_p%i,%i -> u%i,%i AND p%i,%i
                    model.Add(var['u%i,%i' % (r, i)] + var['p%i,%i' % (j, i)] == 2).OnlyEnforceIf(
                        var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)])

                    # __u%i,%i_AND_p%i,%i -> NOT(a%i,%i)
                    model.AddImplication(var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)], var['a%i,%i' % (r, j)].Not())

                    or_list.append(var['__u%i,%i_AND_p%i,%i' % (r, i, j, i)])

            # ar,j
            or_list.append(var['a%i,%i' % (r, j)])

            model.AddBoolOr(or_list).OnlyEnforceIf(var['u%i,%i' % (r, j)])
            model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) >= 1, var['u%i,%i' % (r, j)])
            # for exp in or_list:

            # model.AddImplication(exp, var['u%i,%i' % (r, j)])
            # model.AddImplication(var['u%i,%i' % (r, j)], exp.Not())

            # avoid duplicates
            model.AddImplication(sum(_.GetVarValueMap()[1] for _ in or_list) == 0, var['u%i,%i' % (r, j)].Not())

    # Constraint 10: for a non-leaf node j, exactly one feature is used
    # NOT vj -> (SUM for r=1, k of ar,j = 1)
    for j in range(1, n + 1):
        s = 0
        for r in range(1, k + 1):
            s += var['a%i,%i' % (r, j)]
        model.Add(s == 1).OnlyEnforceIf(var['v%i' % j].Not())

    # Constraint 11: for a leaf node j, no feature is used
    # vj -> (SUM for r=1, k of ar,j = 0)
    for j in range(1, n + 1):
        s = 0
        for r in range(1, k + 1):
            s += var['a%i,%i' % (r, j)]
        model.Add(s == 0).OnlyEnforceIf(var['v%i' % j])

    # Constraint 12: any positive example must be discriminated if the leaf
    # node is associated with the negative class.
    # vj AND NOT cj -> OR for r=1, k of d*r,j
    # * = 0 if current training example's feature value is 0
    # * = 1 if current training example's feature value is 1
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
    # node is associated with the positive class.
    # vj AND cj -> OR for r=1, k of d*r,j
    # * = 0 if current training example's feature value is 0
    # * = 1 if current training example's feature value is 1
    for example in neg_x:
        for j in range(1, n + 1):
            or_list = []
            for r in range(1, k + 1):
                if example[r - 1] == 0:
                    or_list.append(var['d0%i,%i' % (r, j)])
                else:
                    or_list.append(var['d1%i,%i' % (r, j)])
            model.AddBoolOr(or_list).OnlyEnforceIf([var['v%i' % j], var['c%i' % j]])

    # Constraint 13.1: only a leaf node can be associated to a class.
    # ci -> vi, i=1,..,n
    for i in range(1, n + 1):
        model.AddImplication(var['c%i' % i], var['v%i' % i])
    #'''
    # Additional constraint 1
    for i in range(1, n + 1):
        var['_lambda%i,%i' % (0, i)] = model.NewBoolVar('_lambda%i,%i' % (0, i))
        model.Add(var['_lambda%i,%i' % (0, i)] == 1)

        var['_tau%i,%i' % (0, i)] = model.NewBoolVar('_tau%i,%i' % (0, i))
        model.Add(var['_tau%i,%i' % (0, i)] == 1)

        for t in range(0, n + 1):
            var['_lambda%i,%i' % (t, i)] = model.NewBoolVar('_lambda%i,%i' % (t, i))
            var['_tau%i,%i' % (t, i)] = model.NewBoolVar('_tau%i,%i' % (t, i))

    for i in range(1, n + 1):
        for t in range(1, floor(i / 2) + 1):
            if i > 1:
                var['__lambda%i,%i_AND_v%i' % (t - 1, i - 1, i)] = model.NewBoolVar(
                    '__lambda%i,%i_AND_v%i' % (t - 1, i - 1, i))

                model.Add(var['_lambda%i,%i' % (t - 1, i - 1)] + var['v%i' % i] == 2).OnlyEnforceIf(
                    var['__lambda%i,%i_AND_v%i' % (t - 1, i - 1, i)])

                model.AddBoolOr([var['_lambda%i,%i' % (t, i - 1)],
                                 var['__lambda%i,%i_AND_v%i' % (t - 1, i - 1, i)]]).OnlyEnforceIf(
                    var['_lambda%i,%i' % (t, i)])

                model.AddImplication(var['_lambda%i,%i' % (t, i - 1)], var['_lambda%i,%i' % (t, i)])
                model.AddImplication(var['__lambda%i,%i_AND_v%i' % (t - 1, i - 1, i)], var['_lambda%i,%i' % (t, i)])

        for t in range(1, i + 1):
            if i > 1:
                var['__tau%i,%i_AND_NOT_v%i' % (t - 1, i - 1, i)] = model.NewBoolVar(
                    '__tau%i,%i_AND_NOT_v%i' % (t - 1, i - 1, i))

                model.Add(var['_tau%i,%i' % (t - 1, i - 1)] + var['v%i' % i].Not() == 2).OnlyEnforceIf(
                    var['__tau%i,%i_AND_NOT_v%i' % (t - 1, i - 1, i)])

                model.AddBoolOr([var['_tau%i,%i' % (t, i - 1)],
                                 var['__tau%i,%i_AND_NOT_v%i' % (t - 1, i - 1, i)]]).OnlyEnforceIf(
                    var['_tau%i,%i' % (t, i)])

                model.AddImplication(var['_tau%i,%i' % (t, i - 1)], var['_tau%i,%i' % (t, i)])
                model.AddImplication(var['__tau%i,%i_AND_NOT_v%i' % (t - 1, i - 1, i)], var['_tau%i,%i' % (t, i)])

    # Additional constraint 2
    for i in range(1, n + 1):
        for t in range(1, floor(i / 2) + 1):
            if 'l%i,%i' % (i, 2 * (i - t + 1)) in var and 'r%i,%i' % (i, 2 * (i - t + 1) + 1) in var:
                model.Add(var['l%i,%i' % (i, 2 * (i - t + 1))] == 0).OnlyEnforceIf(var['_lambda%i,%i' % (t, i)])
                model.Add(var['r%i,%i' % (i, 2 * (i - t + 1) + 1)] == 0).OnlyEnforceIf(var['_lambda%i,%i' % (t, i)])

        for t in range(ceil(i / 2), i + 1):
            if 'l%i,%i' % (i, 2 * (t - 1)) in var and 'r%i,%i' % (i, 2 * (t - 1)) in var:
                model.Add(var['l%i,%i' % (i, 2 * (t - 1))] == 0).OnlyEnforceIf(var['_tau%i,%i' % (t, i)])
                model.Add(var['r%i,%i' % (i, 2 * t - 1)] == 0).OnlyEnforceIf(var['_tau%i,%i' % (t, i)])
    #'''


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


# solver = cp_model.CpSolver()
# solution_printer = VarArraySolutionPrinter(var.values())
# status = solver.SearchForAllSolutions(model, solution_printer)

# print('Status = %s' % solver.StatusName(status))
# print('Number of solutions found: %i' % solution_printer.solution_count())


# Store all the solutions in a list
class VarArraySolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collect intermediate solutions."""

    def __init__(self, variables, limit=1):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.solution_list = []
        self.solution_count = 0
        self.solution_limit = limit

    def on_solution_callback(self):
        """
        Store all the solutions <= limit in a list of dictionaries,
        where each one represents a solution.
        A solution, in turn,  is made up by five dictionaries, respectively
        for the variables v, l, r, a and c.
        """

        self.solution_count += 1

        v_var = {}
        l_var = {}
        r_var = {}
        a_var = {}
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
                elif str(v).startswith('a') and self.Value(v) == 1:
                    feature = int(str(v)[1:].partition(',')[0])
                    node = int(str(v)[1:].partition(',')[2])
                    a_var[node] = feature
                elif str(v).startswith('c') and self.Value(v) == 1:
                    c_var[int(str(v)[1:])] = self.Value(v)

            except Exception as e:
                traceback.print_exc()
                raise e

        solution = {'v': v_var, 'l': l_var, 'r': r_var, 'a': a_var, 'c': c_var}

        # Check for duplicate solutions
        if solution not in self.solution_list:
            self.solution_list.append(solution)

        # Stop the search after reaching the limit
        if self.solution_count >= self.solution_limit:
            self.StopSearch()


def get_solution(x_values, y_values, target_nodes):
    """ Returns all the possible solutions, or an empty tuple if no solution is found."""

    n = target_nodes  # number of nodes

    # select only the rows where the target feature equals 1
    pos_x = x_values[y_values.astype(np.bool), :]

    # select only the rows where the target feature equals 0
    neg_x = x_values[~y_values.astype(np.bool), :]

    k = len(x_values[0])

    set_csp(pos_x, neg_x, n, k)

    # Create a solver and solve the model.
    solver = cp_model.CpSolver()

    solver.parameters.num_search_workers = 8

    # print log during search, useful for debug
    # solver.parameters.log_search_progress = True

    global model
    status = solver.Solve(model)

    solution = None

    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:

        global var

        v_var = {}
        l_var = {}
        r_var = {}
        a_var = {}
        c_var = {}

        for k, v in var.items():
            try:
                if k.startswith('v'):
                    v_var[int(k[1:])] = solver.Value(v)
                elif k.startswith('l') and solver.Value(v) == 1:
                    parent = k[1:].partition(',')[0]
                    child = k[1:].partition(',')[2]
                    l_var[int(parent)] = int(child)
                elif k.startswith('r') and solver.Value(v) == 1:
                    parent = k[1:].partition(',')[0]
                    child = k[1:].partition(',')[2]
                    r_var[int(parent)] = int(child)
                elif k.startswith('a') and solver.Value(v) == 1:
                    feature = k[1:].partition(',')[0]
                    node = k[1:].partition(',')[2]
                    a_var[int(node)] = int(feature)
                elif k.startswith('c') and solver.Value(v) == 1:
                    c_var[int(k[1:])] = solver.Value(v)

            except Exception as e:
                traceback.print_exc()
                raise e

        solution = {'v': v_var, 'l': l_var, 'r': r_var, 'a': a_var, 'c': c_var}

    return solution
