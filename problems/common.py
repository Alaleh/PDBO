import numpy as np
from problems import *
from external import lhs
from pymoo.factory import get_from_list, get_reference_directions


def get_problem_options():
    problems = [
        ('zdt1', ZDT1),
        ('zdt2', ZDT2),
        ('zdt3', ZDT3),
        ('dtlz1', DTLZ1),
        ('dtlz3', DTLZ3),
        ('dtlz5', DTLZ5),
        ('gtd', re.GearTrainDesign),
    ]
    return problems


def get_problem(name, *args, d={}, **kwargs):
    return get_from_list(get_problem_options(), name.lower(), args, {**d, **kwargs})


def generate_initial_samples(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    X_feasible = np.zeros((0, problem.n_var))
    Y_feasible = np.zeros((0, problem.n_obj))

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    while len(X_feasible) < n_sample:
        X = lhs(problem.n_var, n_sample)
        X = problem.xl + X * (problem.xu - problem.xl)
        Y, feasible = problem.evaluate(X, return_values_of=['F', 'feasible'])
        feasible = feasible.flatten()
        X_feasible = np.vstack([X_feasible, X[feasible]])
        Y_feasible = np.vstack([Y_feasible, Y[feasible]])
    
    indices = np.random.permutation(np.arange(len(X_feasible)))[:n_sample]
    X, Y = X_feasible[indices], Y_feasible[indices]
    return X, Y


def  build_problem(name, n_var, n_obj, n_init_sample, n_process=1):
    '''
    Build optimization problem from name, get initial samples
    Input:
        name: name of the problem (supports ZDT1-6, DTLZ1-7)
        n_var: number of design variables
        n_obj: number of objectives
        n_init_sample: number of initial samples
        n_process: number of parallel processes
    Output:
        problem: the optimization problem
        X_init, Y_init: initial samples
        pareto_front: the true pareto front of the problem (if defined, otherwise None)
    '''
    # build problem
    if name.startswith('zdt'):
        problem = get_problem(name, n_var=n_var)
    elif name.startswith('dtlz'):
        problem = get_problem(name, n_var=n_var, n_obj=n_obj)
    else:
        try:
            problem = get_problem(name)
        except:
            raise NotImplementedError('problem not supported yet!')

    # get initial samples
    X_init, Y_init = generate_initial_samples(problem, n_init_sample)
    
    return problem, [], X_init, Y_init