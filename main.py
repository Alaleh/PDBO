"""
The code structure is inspired from the implementation of [1], we thank the authors for their code.
[1] Konakovic Lukovic, Mina, Yunsheng Tian, and Wojciech Matusik. "Diversity-guided multi-objective bayesian optimization with batch evaluations." Advances in Neural Information Processing Systems 33 (2020): 17708-17720.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'  # speed up
import numpy as np
from arguments import get_args
from baselines.vis_data_export import DataExport
from problems.common import build_problem
from mobo.algorithms import get_algorithm
from utils import save_args, setup_logger


def main():
    args, framework_args = get_args()

    np.random.seed(args.seed)
    problem, true_pfront, X_init, Y_init = build_problem(args.problem, args.n_var, args.n_obj, args.n_init_sample, args.n_process)
    args.n_var, args.n_obj = problem.n_var, problem.n_obj

    optimizer = get_algorithm(args.algo)(problem, args.n_iter, args.ref_point, framework_args)

    save_args(args, framework_args)
    logger = setup_logger(args)
    print(problem, optimizer, sep='\n')

    exporter = DataExport(optimizer, X_init, Y_init, args)
    solution = optimizer.solve(X_init, Y_init)

    for _ in range(args.n_iter):
        X_next, Y_next = next(solution)
        exporter.update(X_next, Y_next)
        exporter.write_csvs()

    if logger is not None:
        logger.close()


if __name__ == '__main__':
    main()