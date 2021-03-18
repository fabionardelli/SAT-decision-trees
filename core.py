# This software implements a method for learning a decision tree as a SAT problem,
# as shown in the paper "Learning Optimal Decision Trees with SAT" by Nina Narodytska et al., 2018
# All credit for the encoding proposed here must be given to the authors of the paper.
# Author: Fabio Nardelli

import numpy as np
from yices import *
from math import floor, ceil
import traceback
import pandas as pd
import itertools

# Global variables to be used later on.
var = {}  # to store the variables of the csp

cfg = None
ctx = None


def set_csp(pos_x, neg_x, n, k):
    """Creates the variables and the constraints for a given number of nodes."""

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

    bool_t = Types.bool_type()

    # Create a variable 'v' for each node in the tree
    # It is True iff node i is a leaf node
    for i in range(1, n + 1):
        var['v%i' % i] = Terms.new_uninterpreted_term(bool_t, 'v%i' % i)

    # Create a variable 'l' for each possible left child of current node
    # It is True iff node i has node j as left child
    for i in range(1, n + 1):
        for j in get_lr(i):
            var['l%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'l%i,%i' % (i, j))

    # Create a variable 'r' for each possible right child of current node
    # It is True iff node i has node j as right child
    for i in range(1, n + 1):
        for j in get_rr(i):
            var['r%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'r%i,%i' % (i, j))

    # Create a variable 'p' to tell the parent of current node
    # It is True iff the parent of node j is node i
    for i in range(1, n):
        for j in range(i + 1, min(2 * i + 1, n) + 1):
            var['p%i,%i' % (j, i)] = Terms.new_uninterpreted_term(bool_t, 'p%i,%i' % (j, i))

    # Create a variable 'a' for each couple (node, feature)
    # It is True iff feature r is assigned to node j
    for i in range(1, k + 1):
        for j in range(1, n + 1):
            var['a%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'a%i,%i' % (i, j))

    # Create a variable 'u' for each couple (node, feature)
    # It is True iff feature r is being discriminated against by node j
    for i in range(1, k + 1):
        for j in range(1, n + 1):
            var['u%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'u%i,%i' % (i, j))

    # Create a variable 'd0' for each couple (node, feature)
    # It is True iff feature r is discriminated for value 0 by node j
    # or by one of its ancestors
    for i in range(1, k + 1):
        for j in range(1, n + 1):
            var['d0%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'd0%i,%i' % (i, j))

    # Create a variable 'd1' for each couple (node, feature)
    # It is True iff feature r is discriminated for value 1 by node j
    # or by one of its ancestors
    for i in range(1, k + 1):
        for j in range(1, n + 1):
            var['d1%i,%i' % (i, j)] = Terms.new_uninterpreted_term(bool_t, 'd1%i,%i' % (i, j))
    
    # Create a variable 'c' for each node.
    # It is True iff class of leaf node j is 1
    for j in range(1, n + 1):
        var['c%i' % j] = Terms.new_uninterpreted_term(bool_t, 'c%i' % j)

    # Constraints.

    global cfg
    cfg = Config()
    cfg.default_config_for_logic('QF_LRA')

    global ctx
    ctx = Context(cfg)

    # enable multithreading

    # TREE BUILDING CONSTRAINTS
    # These constraints allow to represent a full binary tree, that is a tree
    # in which each node has exactly 0 or 2 children. Since the tree cannot be
    # made up by the root alone, the number of nodes 'n' must be an odd number >= 3.
    # Given n, the constraints permit to find all the possible full binary trees
    # which can be built with n nodes. The number of legit trees with n nodes
    # is given by the Catalan numbers sequence.

    # Constraint 1: the root node is not a leaf.
    # NOT v1
    ctx.assert_formula(Terms.ynot(var['v1']))

    # Constraint 2: if a node is a leaf node, then it has no children.
    # vi -> NOT li,j, j in LR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            ctx.assert_formula(Terms.implies(var['v%i' % i], Terms.ynot(var['l%i,%i' % (i, j)])))

    # Constraint 3: the left child and the right child of the i-th node
    # are numbered consecutively.
    # li,j <-> ri,j+1, j in LR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            ctx.assert_formula(Terms.implies(var['l%i,%i' % (i, j)], var['r%i,%i' % (i, j + 1)]))
            ctx.assert_formula(Terms.implies(var['r%i,%i' % (i, j + 1)], var['l%i,%i' % (i, j)]))

    # Constraint 4: a non-leaf node must have a child.
    # NOT vi -> (SUM for j in LR(i) of li,j = 1)
    for i in range(1, n + 1):
        sum_list = []
        for j in get_lr(i):
            sum_list.append(var['l%i,%i' % (i, j)])

        exp1 = Terms.yor(sum_list)  # at least one TRUE

        or_list = []
        for pair in itertools.combinations(sum_list, 2):
            # NOT li,j OR NOT lk,h for each pair
            or_list.append(Terms.yor([Terms.ynot(pair[0]), Terms.ynot(pair[1])]))

        # at least one FALSE
        exp2 = Terms.yand(or_list)
        exp = Terms.yand([exp1, exp2])  # exactly one TRUE

        ctx.assert_formula(Terms.implies(Terms.ynot(var['v%i' % i]), exp))
    '''
    # Constraint 4bis: each left/right child must have exactly a parent
    for j in range(2, n + 1):
        left_list = []
        right_list = []
        for i in range(1, n):
            if 'l%i,%i' % (i, j) in var:
                left_list.append(var['l%i,%i' % (i, j)])
            if 'r%i,%i' % (i, j) in var:
                right_list.append(var['r%i,%i' % (i, j)])
            if len(left_list) > 0:
                exp1 = Terms.yor(left_list)  # at least one TRUE

                or_list = []
                for pair in itertools.combinations(left_list, 2):
                    # NOT li,j OR NOT lk,h for each pair
                    or_list.append(Terms.yor([Terms.ynot(pair[0]), Terms.ynot(pair[1])]))

                # at least one FALSE
                exp2 = Terms.yand(or_list)
                ctx.assert_formula(Terms.yand([exp1, exp2]))  # exactly one TRUE

            if len(right_list) > 0:
                exp1 = Terms.yor(right_list)  # at least one TRUE

                or_list = []
                for pair in itertools.combinations(right_list, 2):
                    # NOT li,j OR NOT lk,h for each pair
                    or_list.append(Terms.yor([Terms.ynot(pair[0]), Terms.ynot(pair[1])]))

                # at least one FALSE
                exp2 = Terms.yand(or_list)
                ctx.assert_formula(Terms.yand([exp1, exp2]))  # exactly one TRUE
    #'''
    #'''
    # Constraint 4ter: nodes on the same level must be labeled increasingly
    # li,j -> lh,(j-2), and ri,j -> rh,(j-2), h < i
    for i in range(n - 2, 0, -1):
        for j in reversed(get_lr(i)):
            if 'l%i,%i' % (i, j) in var:
                node_list = []
                for h in range(i - 1, 0, -1):
                    if 'l%i,%i' % (h, j - 2) in var:
                        node_list.append(var['l%i,%i' % (h, j - 2)])
                if len(node_list) > 0:
                    exp = Terms.yor(node_list)
                    ctx.assert_formula(Terms.implies(var['l%i,%i' % (i, j)], exp))

        for j in reversed(get_rr(i)):
            if 'r%i,%i' % (i, j) in var:
                node_list = []
                for h in range(i - 1, 0, -1):
                    if 'r%i,%i' % (h, j - 2) in var:
                        node_list.append(var['r%i,%i' % (h, j - 2)])
                if len(node_list) > 0:
                    exp = Terms.yor(node_list)
                    ctx.assert_formula(Terms.implies(var['r%i,%i' % (i, j)], exp))
    #'''
    # Constraint 5: if the i-th node is a parent then it must have a child
    # pj,i <-> li,j, j in LR(i)
    # pj,i <-> ri,j, j in RR(i)
    for i in range(1, n + 1):
        for j in get_lr(i):
            ctx.assert_formula(Terms.implies(var['p%i,%i' % (j, i)], var['l%i,%i' % (i, j)]))
            ctx.assert_formula(Terms.implies(var['l%i,%i' % (i, j)], var['p%i,%i' % (j, i)]))
        for j in get_rr(i):
            ctx.assert_formula(Terms.implies(var['p%i,%i' % (j, i)], var['r%i,%i' % (i, j)]))
            ctx.assert_formula(Terms.implies(var['r%i,%i' % (i, j)], var['p%i,%i' % (j, i)]))

    # Constraint 6: all the nodes but the first must have a parent.
    # (SUM for i=floor(j/2), min(j-1, N) of pj,i = 1), j =2,...,n
    for j in range(2, n + 1):
        sum_list = []
        for i in range(floor(j / 2), min(j - 1, n) + 1):
            sum_list.append(var['p%i,%i' % (j, i)])

        exp1 = Terms.yor(sum_list)  # at least one TRUE

        or_list = []
        for pair in itertools.combinations(sum_list, 2):
            # NOT li,j OR NOT lk,h for each pair
            or_list.append(Terms.yor([Terms.ynot(pair[0]), Terms.ynot(pair[1])]))

        # at least one FALSE
        exp2 = Terms.yand(or_list)
        exp = Terms.yand([exp1, exp2])  # exactly one TRUE

        ctx.assert_formula(exp)

    # LEARNING CONSTRAINTS
    # These constraints allow to learn a decision tree starting from a
    # dataset of binary features. The resulting tree is represented
    # as a total assignment to the variables. The values of these variables
    # must be used to build a tree and implement the classifier.

    # Constraint 7: to discriminate a feature for value 0 at node j = 2,...,n
    # d0r,j <-> (OR for i=floor(j/2), j-1 of ((pj,i AND d0r,i) OR (ar,i AND ri,j)))
    # d0r,1 = 0
    for r in range(1, k + 1):
        ctx.assert_formula(Terms.ynot(var['d0%i,%i' % (r, 1)]))  # d0r,1 = 0

        for j in range(2, n + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1 and (j in get_lr(i) or j in get_rr(i)) and 'r%i,%i' % (i, j) in var:
                    or_list.append(Terms.yor([Terms.yand([var['p%i,%i' % (j, i)], var['d0%i,%i' % (r, i)]]),
                                              Terms.yand([var['a%i,%i' % (r, i)], var['r%i,%i' % (i, j)]])]))
            ctx.assert_formula(Terms.implies(var['d0%i,%i' % (r, j)], Terms.yor(or_list)))
            ctx.assert_formula(Terms.implies(Terms.yor(or_list), var['d0%i,%i' % (r, j)]))

    # Constraint 8: to discriminate a feature for value 1 at node j = 2,...,n
    # d1r,j <-> (OR for i=floor(j/2), j-1 of ((pj,i AND d1r,i) OR (ar,i AND li,j)))
    # d1r,1 = 0
    for r in range(1, k + 1):
        ctx.assert_formula(Terms.ynot(var['d1%i,%i' % (r, 1)]))  # d1r,1 = 0

        for j in range(2, n + 1):
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1 and (j in get_lr(i) or j in get_rr(i)) and 'l%i,%i' % (i, j) in var:
                    or_list.append(Terms.yor([Terms.yand([var['p%i,%i' % (j, i)], var['d1%i,%i' % (r, i)]]),
                                              Terms.yand([var['a%i,%i' % (r, i)], var['l%i,%i' % (i, j)]])]))
            ctx.assert_formula(Terms.implies(var['d1%i,%i' % (r, j)], Terms.yor(or_list)))
            ctx.assert_formula(Terms.implies(Terms.yor(or_list), var['d1%i,%i' % (r, j)]))

    # Constraint 9: using a feature r at node j, r = 1,...,k, j = 1,...,n
    # AND for i = floor(j/2), j-1 of (ur,i ^ pj,i -> -ar,j)
    # ur,j <-> (ar,j OR (OR for i = floor(j/2), j-1 of (ur,j AND pj,i)))
    for r in range(1, k + 1):
        for j in range(1, n + 1):
            and_list = []
            or_list = []
            for i in range(floor(j / 2), j):
                if i >= 1:  # and j in getLR(i) or j in getRR(i):
                    # ur,i ^ pj,i -> -ar,j
                    and_list.append(Terms.implies(Terms.yand([var['u%i,%i' % (r, i)], var['p%i,%i' % (j, i)]]),
                                                  Terms.ynot(var['a%i,%i' % (r, j)])))
                    # AND for i = floor(j/2), j-1 of (ur,i ^ pj,i -> -ar,j)
                    ctx.assert_formula(Terms.yand(and_list))

                    or_list.append(Terms.yand([var['u%i,%i' % (r, i)], var['p%i,%i' % (j, i)]]))

            ctx.assert_formula(Terms.implies(var['u%i,%i' % (r, j)], Terms.yor([var['a%i,%i' % (r, j)], *or_list])))
            ctx.assert_formula(Terms.implies(Terms.yor([var['a%i,%i' % (r, j)], *or_list]), var['u%i,%i' % (r, j)]))

    # Constraint 10: for a non-leaf node j, exactly one feature is used
    # NOT vj -> (SUM for r=1, k of ar,j = 1)
    for j in range(1, n + 1):
        sum_list = []
        for r in range(1, k + 1):
            sum_list.append(var['a%i,%i' % (r, j)])

        exp1 = Terms.yor(sum_list)  # at least one TRUE

        or_list = []
        for pair in itertools.combinations(sum_list, 2):
            # NOT li,j OR NOT lk,h for each pair
            or_list.append(Terms.yor([Terms.ynot(pair[0]), Terms.ynot(pair[1])]))

        # at least one FALSE
        exp2 = Terms.yand(or_list)
        exp = Terms.yand([exp1, exp2])  # exactly one TRUE

        ctx.assert_formula(Terms.implies(Terms.ynot(var['v%i' % j]), exp))

    # Constraint 11: for a leaf node j, no feature is used
    # vj -> (SUM for r=1, k of ar,j = 0)
    for j in range(1, n + 1):
        sum_list = []
        for r in range(1, k + 1):
            sum_list.append(var['a%i,%i' % (r, j)])

        exp = Terms.ynot(Terms.yor(sum_list))  # all FALSE

        ctx.assert_formula(Terms.implies(var['v%i' % j], exp))

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

            ctx.assert_formula(
                Terms.implies(Terms.yand([var['v%i' % j], Terms.ynot(var['c%i' % j])]), Terms.yor(or_list)))

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

            ctx.assert_formula(Terms.implies(Terms.yand([var['v%i' % j], var['c%i' % j]]), Terms.yor(or_list)))

    # Constraint 13-bis: only a leaf node can be associated to a class.
    # ci -> vi, i=1,..,n
    for i in range(1, n + 1):
        ctx.assert_formula(Terms.implies(var['c%i' % i], var['v%i' % i]))

    #'''
    # Additional constraint 1
    for i in range(1, n + 1):
        for t in range(0, n + 1):
            var['_lambda%i,%i' % (t, i)] = Terms.new_uninterpreted_term(bool_t, '_lambda%i,%i' % (t, i))
            var['_tau%i,%i' % (t, i)] = Terms.new_uninterpreted_term(bool_t, '_tau%i,%i' % (t, i))

            # lambda0,i = 1, tau0,i = 1
            if t == 0:
                ctx.assert_formula(var['_lambda%i,%i' % (t, i)])
                ctx.assert_formula(var['_tau%i,%i' % (t, i)])

    for i in range(1, n + 1):
        for t in range(1, floor(i / 2) + 1):
            if i > 1:
                # lambda t,i -> (lambda t,i-1 OR lambda t-1,i-1 AND vi)
                ctx.assert_formula(Terms.implies(var['_lambda%i,%i' % (t, i)], Terms.yor(
                    [var['_lambda%i,%i' % (t, i - 1)],
                     Terms.yand([var['_lambda%i,%i' % (t - 1, i - 1)], var['v%i' % i]])])))
                # (lambda t,i-1 OR lambda t-1,i-1 AND vi) -> lambda t,i
                ctx.assert_formula(Terms.implies(Terms.yor([var['_lambda%i,%i' % (t, i - 1)], Terms.yand(
                    [var['_lambda%i,%i' % (t - 1, i - 1)], var['v%i' % i]])]), var['_lambda%i,%i' % (t, i)]))

        for t in range(1, i + 1):
            if i > 1:
                # tau t,i -> (tau t,i-1 OR tau t-1,i-1 AND vi)
                ctx.assert_formula(Terms.implies(var['_tau%i,%i' % (t, i)], Terms.yor([var['_tau%i,%i' % (t, i - 1)],
                                                                                       Terms.yand([var['_tau%i,%i' % (
                                                                                           t - 1, i - 1)], Terms.ynot(
                                                                                           var['v%i' % i])])])))
                # (tau t,i-1 OR tau t-1,i-1 AND vi) -> tau t,i
                ctx.assert_formula(Terms.implies(Terms.yor([var['_tau%i,%i' % (t, i - 1)], Terms.yand(
                    [var['_tau%i,%i' % (t - 1, i - 1)], Terms.ynot(var['v%i' % i])])]), var['_tau%i,%i' % (t, i)]))

    # Additional constraint 2
    for i in range(1, n + 1):
        for t in range(1, floor(i / 2) + 1):
            if 'l%i,%i' % (i, 2 * (i - t + 1)) in var and 'r%i,%i' % (i, 2 * (i - t + 1) + 1) in var:
                ctx.assert_formula(Terms.implies(var['_lambda%i,%i' % (t, i)], Terms.yand(
                    [Terms.ynot(var['l%i,%i' % (i, 2 * (i - t + 1))]),
                     Terms.ynot(var['r%i,%i' % (i, 2 * (i - t + 1) + 1)])])))

        for t in range(ceil(i / 2), i + 1):
            if 'l%i,%i' % (i, 2 * (t - 1)) in var and 'r%i,%i' % (i, 2 * (t - 1)) in var:
                ctx.assert_formula(Terms.implies(var['_tau%i,%i' % (t, i)], Terms.yand(
                    [Terms.ynot(var['l%i,%i' % (i, 2 * (t - 1))]), Terms.ynot(var['r%i,%i' % (i, 2 * (t - 1))])])))
    #'''


def get_solution(x_values, y_values, target_nodes, time_limit=1000):
    """ Returns all the possible solutions, or an empty tuple if no solution is found."""

    n = target_nodes  # number of nodes

    # select only the rows where the target feature equals 1
    pos_x = x_values[y_values.astype(np.bool), :]

    # select only the rows where the target feature equals 0
    neg_x = x_values[~y_values.astype(np.bool), :]

    k = len(x_values[0])

    set_csp(pos_x, neg_x, n, k)

    global ctx
    status = ctx.check_context()

    if status == Status.SAT:

        m = Model.from_context(ctx, 1)

        v_var = {}
        l_var = {}
        r_var = {}
        a_var = {}
        c_var = {}

        global var

        for k, v in var.items():
            try:
                if k.startswith('v'):
                    v_var[int(k[1:])] = m.get_bool_value(v)
                elif k.startswith('l') and m.get_bool_value(v) == 1:
                    parent = k[1:].partition(',')[0]
                    child = k[1:].partition(',')[2]
                    l_var[int(parent)] = int(child)
                elif k.startswith('r') and m.get_bool_value(v) == 1:
                    parent = k[1:].partition(',')[0]
                    child = k[1:].partition(',')[2]
                    r_var[int(parent)] = int(child)
                elif k.startswith('a') and m.get_bool_value(v) == 1:
                    feature = k[1:].partition(',')[0]
                    node = k[1:].partition(',')[2]
                    a_var[int(node)] = int(feature)
                elif k.startswith('c') and m.get_bool_value(v) == 1:
                    c_var[int(k[1:])] = m.get_bool_value(v)

            except Exception as e:
                traceback.print_exc()
                raise e

        solution = {'v': v_var, 'l': l_var, 'r': r_var, 'a': a_var, 'c': c_var}

        ctx.dispose()
        cfg.dispose()

        return solution
    return None


'''
data = pd.read_csv('data.csv', delimiter=',', header=None, skiprows=1)
# data = pd.read_csv('datasets/binary/bin-weather.csv', delimiter=',', header=None)
X = data.iloc[:, :-1].to_numpy(dtype=np.int8)
y = data.iloc[:, -1].to_numpy(dtype=np.int8)

sol = get_solution(X, y, 5)

print(sol)

vd = sol.get('v')
for k, v in vd.items():
    print(k, v)
'''
