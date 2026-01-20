import math
import os
import sys

import cplex
import numpy as np
from pysat.examples.hitman import Hitman

from explaining.encoding_utils_functions import generate_variables, separate_vars, get_max, get_support_variables_names


def define_number_of_outputs(model):
    inpt = []

    for layer_name in ["firstlayer", "secondlayer"]:
        inpt.append(model.get_layer(layer_name).input_shape[1])

    return inpt


def get_weights_and_bias(model, n_layers):
    A = []
    b = []

    for n_of_layer in range(n_layers):
        total_weights = model.layers[n_of_layer].get_weights()

        weights = total_weights[0]
        bias = total_weights[1]

        temp_layer_weights = []
        temp_bias_layer = []

        w = weights.transpose()

        for i in range(len(w)):
            temp_layer_weights.append(w[i])
            temp_bias_layer.append(bias[i])

        b.append(temp_bias_layer)
        A.append(temp_layer_weights)

    return A, b


def read_categorical_indexes(df_name):
    sep = os.sep
    path = "." + sep + "datasets_categorical_index" + sep + df_name + sep + df_name + "_categorical_indexes.txt"

    ids = []
    if os.path.exists(path):
        with open(path) as f:
            lines = f.read().splitlines()
        for v in lines:
            ids.append([int(e) for e in v.split("_")])

    return ids


def set_indicator_constraint(formula, support_variables):
    import re
    form = [int(v) for v in re.findall(r'\d+', support_variables[0])]
    var_id, n_layer = form[0], form[1]
    z_var = "Z_{0}_layer_{1}".format(var_id, n_layer)
    formula.variables.add(names=[z_var], types=["B"])
    idx = formula.variables.get_indices(support_variables)
    first = True
    for i in idx:
        flag = 0 if first else 1; first = False

        formula.indicator_constraints.add(indvar=z_var,
                                          lin_expr=[[i], [1]],
                                          sense="L",
                                          rhs=0,
                                          complemented=flag,
                                          name="ic_{0}_layer_{1}".format(var_id, n_layer),
                                          indtype=formula.indicator_constraints.type_.if_)


def get_vars(formula, name):
    formula_variables = [v for v in formula.variables.get_names() if name[0] in v] if len(name) == 1 \
        else [v for v in formula.variables.get_names() if name[0] in v and name[1] in v]

    return formula_variables


def drop_elements_from_list(initial, to_remove):
    return [v for v in initial if v not in to_remove]


def decouple_couple(var_index, weights):
    return [list(value) for value in zip(*[couple for couple in zip(var_index, weights)])]


def freeze_input_and_output_CPLEX(pb, inputs, outputs):

    input_vars, input_values = inputs
    output_vars, output_values = outputs

    hypos = freeze_input_CPLEX(pb, input_vars, input_values)
    freeze_output_CPLEX(pb, output_vars, output_values)

    return hypos


def freeze_input_CPLEX(pb, input_vars, input_value):

    hypos = []

    for id_variable in range(len(input_vars)):
        var = input_vars[id_variable]
        val = input_value[id_variable]

        eql, rhs = [[var], [1]], [float(val)]

        cnames = ['hypo_{0}'.format(id_variable)]
        senses = ['E']
        constr = [eql]

        pb.linear_constraints.add(lin_expr=constr, senses=senses, rhs=rhs, names=cnames)

        # adding a constraint to the list of hypotheses
        hypos.append(tuple([cnames[0], constr, rhs, senses, id_variable]))

    return hypos


def freeze_output_CPLEX(pb, output_vars, output_value):
    # adding indicators for correct and wrong outputs

    pb.variables.add(names=['c_{0}'.format(i) for i in range(len(output_vars))], types='B' * len(output_vars))
    for i in range(len(output_vars)):
        ivar = 'c_{0}'.format(i)
        wrong = ['wc_{0}_{1}'.format(i, j) for j in range(len(output_vars)) if i != j]
        pb.variables.add(names=wrong, types='B' * len(wrong))

        # ivar implies at least one wrong class
        pb.indicator_constraints.add(indvar=ivar, lin_expr=[wrong, [1] * len(wrong)], sense='G', rhs=1)

        for j in range(len(output_vars)):
            if i != j:
                # iv => (o_j - o_i >= 0.0000001)
                iv = 'wc_{0}_{1}'.format(i, j)
                ov, oc = [output_vars[j], output_vars[i]], [1, -1]
                pb.indicator_constraints.add(indvar=iv, lin_expr=[ov, oc], sense='G', rhs=0.0001)

    id_class = output_value.index(1)

    pb.linear_constraints.add(lin_expr=[[['c_{0}'.format(id_class)], [1]]], senses='E', rhs=[1], names=['neg_prediction'])


def contains_variables(variables, all_variables_added):
    return len(variables) > 0 \
           and len(all_variables_added) > 0\
           and len([v for v in variables if v in all_variables_added]) > 0


def define_formula_CPLEX(categorical_ids, A, b):

    n_of_layers = len(A)
    pb = cplex.Cplex()

    pb.set_log_stream(None)
    pb.set_error_stream(None)
    pb.set_warning_stream(None)
    pb.set_results_stream(None)

    all_variables_added = []
    inputs = []
    for n_of_layer in range(n_of_layers):
        w_id = 0

        number_of_vars = len(A[n_of_layer][0])
        variables = generate_variables(n_of_layer, categorical_ids, number_of_vars)

        if n_of_layer == 0:
            inputs = variables.copy()

        if not contains_variables(variables, all_variables_added):
            all_variables_added.extend(variables)
            real, boolean, integers = separate_vars(variables)
            max_values = get_max(integers)

            pb.variables.add(names=integers, lb=[0] * len(integers), ub=max_values, types=["I"] * len(integers))
            pb.variables.add(names=real, types=["C"] * len(real))
            pb.variables.add(names=boolean, types=["B"] * len(boolean))

        for i in range(len(A[n_of_layer])):
            weights = A[n_of_layer][i].tolist()
            if n_of_layer == 0:
                weights.extend([-1, +1])
            else:
                weights.extend([-1])

            rhs = -1 * b[n_of_layer][i]

            support_variables = get_support_variables_names(w_id, n_of_layer)

            if not support_variables[0] in all_variables_added:
                sup_vars_n = len(support_variables)

                if n_of_layer + 1 < 2:
                    pb.variables.add(names=support_variables, lb=[0] * sup_vars_n, types=["C"] * sup_vars_n)
                else:
                    pb.variables.add(names=support_variables, lb=[(float("inf") * -1)] * sup_vars_n, types=["C"] * sup_vars_n)
                all_variables_added.extend(support_variables)

            variables.extend(support_variables)

            var_ids = pb.variables.get_indices(name=variables)
            idx, w = decouple_couple(var_ids, weights)

            pb.linear_constraints.add(lin_expr=[[idx, w]],
                                      rhs=[rhs],
                                      names=["constraint_{0}".format(i)])

            variables = drop_elements_from_list(variables, support_variables)

            if n_of_layer == 0:
                set_indicator_constraint(pb, support_variables)

            w_id += 1

    pb.write("encoded_with_cplex_solver.lp")

    return pb, inputs


def delete_hypos_and_output_constraint(pb):

    vars_to_drop = get_vars(pb, ["c"])

    linear_to_drop = [v for v in pb.linear_constraints.get_names() if "hypo" in v]
    indicator_to_drop = [v for v in pb.indicator_constraints.get_names() if "layer" not in v]

    if linear_to_drop is not []:
        linear_to_drop = list(set(linear_to_drop))
        pb.linear_constraints.delete(linear_to_drop)

    pb.indicator_constraints.delete(indicator_to_drop)
    pb.linear_constraints.delete("neg_prediction")

    pb.variables.delete(vars_to_drop)


def generate_matrix(sample):
    sz = int(math.sqrt(len(sample)))
    # original image
    pixels2 = []
    for i in range(sz):
        row1, row2 = [], []
        for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
            id_pixel = i * sz + j
            row2.append(id_pixel)

        pixels2.append(np.array(row2))

    return np.array(pixels2)


def clockwise_sort(a, key=None, reverse=False):
    nr, nc = a.shape
    res = a.tolist()
    sa = a.ravel().tolist()
    if key is None:
        sa.sort(reverse=reverse)
    else:
        sa.sort(key=key, reverse=reverse)
    res[0] = sa[:nc]
    cur, lenr, lenc = nc, nr - 1, nc - 1
    x, y = 0, nc - 1
    while lenc > 0 and lenr > 0:
        # go down, then go left
        for _ in range(lenr):
            x += 1
            res[x][y] = sa[cur]
            cur += 1
        for _ in range(lenc):
            y -= 1
            res[x][y] = sa[cur]
            cur += 1
        lenr -= 1
        lenc -= 1

        # go up, then go right
        for _ in range(lenr):
            x -= 1
            res[x][y] = sa[cur]
            cur += 1
        for _ in range(lenc):
            y += 1
            res[x][y] = sa[cur]
            cur += 1
        lenr -= 1
        lenc -= 1

    return np.array([k for j in res for k in j])


def clockwise_sort_hypos(hypos):

    orientation = clockwise_sort(generate_matrix([i for i in range(len(hypos))]))

    for i in range(len(hypos)):
        hypos[i] = list(hypos[i])
        hypos[i][4] = orientation[i]
        hypos[i] = tuple(hypos[i])

    hypos.sort(key=lambda x: x[4])

    return hypos


def black_and_white_hypos_sorting(hypos):
    white = [h[4] for h in hypos if h[2][0] == 1]
    hypos_indexes = [h[4] for h in hypos]
    black = set(hypos_indexes).difference(set(white))

    sorted_ids = list(black) + list(white)
    hs = [hypos[sorted_ids[i]] for i in range(len(sorted_ids))]

    return hs


def compute_minimal_CPLEX(oracle, hypos):
    rhypos = []

    # simple deletion-based linear search
    for i, hypo in enumerate(hypos):
        oracle.linear_constraints.delete(hypo[0])

        oracle.solve()
        if oracle.solution.is_primal_feasible():
            # this hypothesis is needed
            # adding it back to the list
            oracle.linear_constraints.add(lin_expr=hypo[1], senses=hypo[3], rhs=hypo[2], names=[hypo[0]])

            rhypos.append(tuple([hypo[4], hypo[0]]))

    return rhypos


def smallest_expl_CPLEX(oracle, hypos):

    def add_hypos(hypos_set):
        for hypothesis in hypos_set:
            oracle.linear_constraints.add(lin_expr=hypos[hypothesis][1], senses=hypos[hypothesis][3],
                                          rhs=hypos[hypothesis][2], names=[hypos[hypothesis][0]])

    def reset_hypos():
        for hypothesis in hypos:
            if hypothesis[0] in oracle.linear_constraints.get_names():
                oracle.linear_constraints.delete(hypothesis[0])

    with Hitman(bootstrap_with=[[i for i in range(len(hypos))]]) as hitman:

        # computing unit-size MCSes
        for i, hypo in enumerate(hypos):
            oracle.linear_constraints.delete(hypo[0])
            oracle.solve()

            if oracle.solution.is_primal_feasible():
                hitman.hit([i])

            reset_hypos()
            add_hypos([i for i in range(len(hypos))])

        iters = 0
        reset_hypos()

        while True:
            hset = hitman.get()

            iters += 1

            print('iter: ', iters)
            print('cand: ', hset)

            add_hypos(hset)
            oracle.solve()

            if oracle.solution.is_primal_feasible():

                to_hit = []
                satisfied, falsified = [], []

                # free vars are not fixed vars: C \ h
                free_variables = list(set(range(len(hypos))).difference(set(hset)))

                model = oracle.solution

                for h in free_variables:

                    var, exp = hypos[h][1][0][0][0], hypos[h][2][0]

                    if "_C" in var:
                        true_val = float(model.get_values(var))
                        add = not (exp - 0.001 <= true_val <= exp + 0.001)
                    else:
                        true_val = int(model.get_values(var))
                        add = exp != true_val

                    if add:
                        falsified.append(h)
                    else:
                        hset.append(h)

                # falsified + satisfied = C \ h

                reset_hypos()
                for u in falsified:
                    to_add = [u] + hset
                    add_hypos(to_add)

                    oracle.solve()

                    if oracle.solution.is_primal_feasible():
                        hset.append(u)
                    else:
                        to_hit.append(u)

                    reset_hypos()

                hitman.hit(to_hit)
            else:
                print("result: ", hset)
                return hset


def minimal_expl_CPLEX(oracle, hypos):

    oracle.solve()
    if oracle.solution.is_primal_feasible():
        print('no implication!')
        values = oracle.solution.get_values()
        names = oracle.variables.get_names()
        for i in range(len(names)):
            print("{0}: {1}".format(names[i], values[i]))
        sys.exit(1)

    return compute_minimal_CPLEX(oracle, hypos)
