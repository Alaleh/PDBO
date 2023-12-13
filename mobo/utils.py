import copy
import numpy as np
from time import time
from pymoo.factory import get_performance_indicator


class Timer:
    def __init__(self):
        self.t = time()

    def log(self, string=None, reset=True):
        msg = '%.2fs' % (time() - self.t)

        if string is not None:
            msg = string + ': ' + msg
        print(msg)
        
        if reset:
            self.t = time()

    def reset(self):
        self.t = time()


def find_pareto_front(Y, return_index=False):
    if len(Y) == 0: return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front


def calc_hypervolume(pfront, ref_point):
    hv = get_performance_indicator('hv', ref_point=ref_point)
    return hv.calc(pfront)


def safe_divide(x1, x2):
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


def expand(x, axis=-1):
    return np.expand_dims(x, axis=axis)


def calculate_HV(front, ref_point):
    pfront = find_pareto_front(front, return_index=False)
    hv_value = calc_hypervolume(pfront, ref_point=ref_point)
    return hv_value


def calculate_contribution_HV(prev_front, added_front, ref_point):
    hv1 = calculate_HV(prev_front, ref_point)
    f_new = copy.deepcopy(prev_front)
    np.append(f_new,added_front)
    hv2 = calculate_HV(f_new, ref_point)
    return hv2 - hv1