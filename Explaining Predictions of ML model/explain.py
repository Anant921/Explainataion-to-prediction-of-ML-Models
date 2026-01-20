import math
import time

import numpy as np
from matplotlib import image as mpimg
import matplotlib.cm as mpcm

from explaining.encoding_functions_SMT import define_formula_SMT, copy_solver, freeze_input_and_output_SMT, \
    minimal_expl_SMT, smallest_expl_SMT

#from explaining.encoding_functions_CPLEX import get_weights_and_bias, read_categorical_indexes, define_formula_CPLEX, \
#    freeze_input_and_output_CPLEX, minimal_expl_CPLEX, delete_hypos_and_output_constraint, smallest_expl_CPLEX




def encoding_model(model, df_name, solver):
    assert solver in ["smt", "cplex"]

    N_OF_LAYERS = 2
    A, b = get_weights_and_bias(model, N_OF_LAYERS)
    categorical_ids = read_categorical_indexes(df_name)

    pb = None
    inputs = []

    if solver == "smt":
        pb, inputs = define_formula_SMT(categorical_ids, A, b)

    elif solver == "cplex":
        pb, inputs = define_formula_CPLEX(categorical_ids, A, b)

    return pb, inputs


def explaining_procedures(pb, expl_type, solver_name, pattern_number, df_name, inputs, outputs):
    explanation = []
    end_time = None
    
    if solver_name == "smt":

        s = copy_solver(pb)
        hypos = freeze_input_and_output_SMT(s, inputs, outputs)

        start_time = time.time()
        explanation = minimal_expl_SMT(s, hypos) if expl_type == "minimal" else smallest_expl_SMT(s, hypos)
        end_time = time.time() - start_time

    elif solver_name == "cplex":
        hypos = freeze_input_and_output_CPLEX(pb, inputs, outputs)

        start_time = time.time()
        explanation = minimal_expl_CPLEX(pb, hypos) if expl_type == "minimal" else smallest_expl_CPLEX(pb, hypos)
        end_time = time.time() - start_time

        delete_hypos_and_output_constraint(pb)


    if df_name == "mnist":
        input_values = inputs[1]
        expl = [v[0] for v in explanation]
        save_images(input_values, expl, pattern_number)

    expl_sz = len(explanation)
    print('  # hypos left:', expl_sz)

    return expl_sz, end_time


def save_images(sample, expl, pattern_id):

    # image size
    sz = int(math.sqrt(len(sample)))

    light_blue_rgba = tuple([0, 255, 255, 230.0])
    white_rgba = tuple([255, 255, 255, 255.0])
    red_rgba = tuple([186, 6, 6, 255.0])
    black_rgba = tuple([0, 0, 0, 255.0])

    # original image
    pixels1, pixels2 = [], []  # this will contain an array of masked pixels
    for i in range(sz):
        row1, row2 = [], []
        for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
            id_pixel = i * sz + j

            if v == 1:
                if id_pixel in expl:
                    row1.append(light_blue_rgba)
                else:
                    row1.append(white_rgba)

                row2.append(white_rgba)
            else:
                if id_pixel in expl:
                    row1.append(red_rgba)
                else:
                    row1.append(black_rgba)

                row2.append(black_rgba)

        pixels1.append(row1)
        pixels2.append(row2)

    pixels1 = np.asarray(pixels1, dtype=np.uint8)
    pixels2 = np.asarray(pixels2, dtype=np.uint8)
    mpimg.imsave('.\\images\\sample{0}-patch.png'.format(pattern_id), pixels1, cmap=mpcm.gray, dpi=5)
    mpimg.imsave('.\\images\\sample{0}-orig.png'.format(pattern_id), pixels2, cmap=mpcm.gray, dpi=5)
