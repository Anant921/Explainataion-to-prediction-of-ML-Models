def check_categorical_id(current_id, categorical_ids):
    return current_id in [v[0] for v in categorical_ids]


def get_min_and_max(current_id, categorical_ids):
    if check_categorical_id(current_id, categorical_ids):
        for v in categorical_ids:
            if v[0] == current_id:
                return [v[1], v[2]]
    else:
        return ["-inf", "+inf"]


def set_type(current_id, categorical_ids, n_layer):
    limits = get_min_and_max(current_id, categorical_ids)
    categorical = limits[0] != "-inf"
    return "B_min_{0}_max_{1}".format(limits[0], limits[1]) if categorical and n_layer == 0 else "C"


def generate_variables(layer, categorical_ids, n_vars):
    return ['X_{0}_layer_{1}_type_{2}'.format(i, layer, set_type(i, categorical_ids, layer)) for i in range(n_vars)]


def get_support_variables_names(current_id, n_of_layer):
    suffix = '_{0}_layer_{1}_type_C'.format(current_id, n_of_layer + 1)
    return ["X" + suffix, "S" + suffix] if n_of_layer == 0 else ["X" + suffix]


def get_max(variables_to_type):
    max_id = -1

    return [int(v.split("_")[max_id]) for v in variables_to_type]


def separate_vars(variables):
    real = []
    boolean = []
    integers = []

    for v in variables:
        last_char = v.split("_")[-1]
        if last_char == "1":
            boolean.append(v)
        elif last_char == "C":
            real.append(v)
        else:
            integers.append(v)

    return real, boolean, integers
