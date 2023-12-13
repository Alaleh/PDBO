import os
import yaml
from multiprocessing import cpu_count
from argparse import ArgumentParser, Namespace


def get_general_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, help='optimization problem')
    parser.add_argument('--n-var', type=int, help='number of design variables')
    parser.add_argument('--n-obj', type=int, help='number of objectives')
    parser.add_argument('--n-init-sample', type=int, default=5, help='number of initial design samples')
    parser.add_argument('--n-iter', type=int, default=20, help='number of optimization iterations')
    parser.add_argument('--ref-point', type=float, nargs='+', default=None, help='hypervolume reference point')
    parser.add_argument('--batch-size', type=int, default=10, help='size of the selected batch in one iteration')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n-seed', type=int, default=1, help='number of random seeds / test runs')
    parser.add_argument('--algo', type=str, help='type of MOO algorithm to use')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder name for storing results')
    parser.add_argument('--exp-name', type=str, default=None, help='custom experiment name')
    parser.add_argument('--log-to-file', default=False, action='store_true', help='log output to file?')
    parser.add_argument('--n-process', type=int, default=cpu_count(), help='number of parallelized processes')
    args, _ = parser.parse_known_args(args)
    return args


def get_surroagte_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--surrogate', type=str, choices=['gp', 'ts'], default='gp', help='type of the surrogate model')
    parser.add_argument('--n-spectral-pts', type=int, default=100, help='number of points for spectral sampling')
    parser.add_argument('--nu', type=int, choices=[1, 3, 5, -1], default=5, help='matern kernel param (integer, -1 means inf)')
    parser.add_argument('--mean-sample', default=False, action='store_true', help='use mean sample when sampling objective functions')
    args, _ = parser.parse_known_args(args)
    return args


def get_acquisition_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--acquisition', type=str, default='adaptive-hedge', help='type of the acquisition function')
    args, _ = parser.parse_known_args(args)
    return args


def get_solver_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--solver', type=str, choices=['nsga2'], default='nsga2', help='type of the MO solver')
    parser.add_argument('--pop-size', type=int, default=100, help='population size')
    parser.add_argument('--n-gen', type=int, default=10, help='number of generations')
    parser.add_argument('--pop-init-method', type=str, choices=['nds', 'random', 'lhs'], default='nds', help='method to init population')
    parser.add_argument('--n-process', type=int, default=cpu_count(), help='number of parallelized processes')
    parser.add_argument('--batch-size', type=int, default=20, help='size of the selected batch in one iteration')
    args, _ = parser.parse_known_args(args)
    return args


def get_selection_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--selection', default='dpp', type=str, help='type of selection method for new batch')
    parser.add_argument('--batch-size', type=int, help='size of the selected batch in one iteration')
    args, _ = parser.parse_known_args(args)
    return args


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--args-path', type=str, default=None, help='used for directly loading arguments from path')
    args, _ = parser.parse_known_args()

    if args.args_path is None:
        general_args = get_general_args()
        surroagte_args = get_surroagte_args()
        acquisition_args = get_acquisition_args()
        solver_args = get_solver_args()
        selection_args = get_selection_args()
        framework_args = {
            'surrogate': vars(surroagte_args),
            'acquisition': vars(acquisition_args),
            'solver': vars(solver_args),
            'selection': vars(selection_args),
        }

    else:
        with open(args.args_path, 'r') as f:
            all_args = yaml.load(f)

        general_args = Namespace(**all_args['general'])
        framework_args = all_args.copy()
        framework_args.pop('general')

    return general_args, framework_args
