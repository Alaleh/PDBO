import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from argparse import ArgumentParser
from problems.common import build_problem


def get_ref_point(problem, n_var, n_obj, n_init_sample=5, seed=0):

    np.random.seed(seed)
    _, _, _, Y_init = build_problem(problem, n_var, n_obj, n_init_sample)

    ref_point_dicts = {'zdt1': [11.0, 11.0], 'zdt2': [11.0, 11.0], 'zdt3': [11.0, 11.0],
                       'dtlz1': [400 for qq in range(n_obj)], 'dtlz3': [10000 for qq in range(n_obj)],
                       'dtlz5': [10.0 for qq in range(n_obj)], 'gtd': [6.6764, 59.0, 0.4633]}
    if problem in ref_point_dicts:
        return ref_point_dicts[problem]

    return np.max(Y_init, axis=0)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, required=True)
    parser.add_argument('--n-var', type=int)
    parser.add_argument('--n-obj', type=int)
    parser.add_argument('--n-init-sample', type=int, default=5)
    args = parser.parse_args()

    ref_point = get_ref_point(args.problem, args.n_var, args.n_obj, args.n_init_sample)

    print(f'Reference point: {ref_point}')