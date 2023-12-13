import os
import copy
import random
import numpy as np
from scipy import optimize
from sklearn import preprocessing
from numpy.random import default_rng
from mobo.PDBO.single_gp import singleGP
from sklearn.gaussian_process.kernels import Sum
from sklearn.gaussian_process.kernels import RBF
from ..utils import calculate_HV, calculate_contribution_HV


def find_batch(batch_size, cheap_points, ground_truth_x, hvs, ground_truth_y, ref_point, alphas=None, alpha_given=False,
               type="dpp-max", kernel_choice="mix"):
    random.seed(1)
    np.random.seed(1)
    GPs = []

    for ii in range(len(ref_point)):
        new_gp = singleGP(dim=len(cheap_points[0]))
        new_gp.fit(ground_truth_x, [pp[ii] for pp in ground_truth_y])
        GPs.append(new_gp)

    if kernel_choice == 'mix':
        kernels_ground_truth = get_kernels(GPs, ground_truth_x)
    else:
        raise "Undefined Kernel"

    if type == 'dpp-max':
        if alphas is None:
            alphas = get_alphas(kernels_ground_truth, hvs)
        points_batch = DPP(GPs, batch_size, alphas, cheap_points, ground_truth_x, hvs, ground_truth_y, ref_point)
        if alpha_given:
            return points_batch, alphas
    else:
        raise "Undefined DPP method"

    return points_batch


def get_kernels(GP_s, points):
    K_s = []
    for g in GP_s:
        g.getCovarianceMatrix(points)
        K_s.append(g.K)
    return K_s


def calculate_matrix_sums(mats, alphas):
    return np.sum(m * a for m, a in zip(mats, alphas))


def mixed_kernel(GPs, alphas):
    m = alphas[0] * GPs[0].gp.kernel_
    for i in range(1, len(alphas)):
        m = Sum(m, alphas[i] * GPs[i].gp.kernel_)
    return m


def get_alphas(kernels, n_hvs):
    norm_hvs = copy.deepcopy(n_hvs)
    norm_hvs = np.asarray(norm_hvs)
    norm_hvs = norm_hvs.reshape(-1, 1)

    def calculate_LML(alpha_i_s):
        if any(alpha_i_s < 0):
            return 10 ** 8 + 44
        k_sums = calculate_matrix_sums(kernels, alpha_i_s)
        covariance = k_sums + np.identity(np.size(k_sums, 0))
        L = np.linalg.cholesky(covariance)
        alph = np.linalg.solve(L.T, np.linalg.solve(L, norm_hvs))
        log_likelihood = -0.5 * norm_hvs.T.dot(alph) - np.sum(np.log(np.diag(L)))
        return log_likelihood[0][0]

    grid = default_rng()
    all_tries = grid.dirichlet(np.ones(len(kernels)), size=10000)
    y_all_tries = [calculate_LML(ss) for ss in all_tries]
    sorted_indecies = np.argsort(y_all_tries)
    best_alphas = all_tries[sorted_indecies[0]]
    y_best = y_all_tries[sorted_indecies[0]]
    new_grid = default_rng()
    opt_bounds = [(0.0, 1.0) for tt in range(len(kernels))]
    opt_constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
    seed_grid = new_grid.dirichlet(np.ones(len(kernels)), size=10)
    for ss in range(len(seed_grid)):
        opt_res = optimize.minimize(calculate_LML, seed_grid[ss], method='trust-constr',
                                    constraints=opt_constraints, bounds=opt_bounds)
        if opt_res != 10 ** 8 + 44 and opt_res['success'] and opt_res['fun'] <= y_best:
            y_best = opt_res['fun']
            best_alphas = opt_res['x']
            break
    print("optimization_results (alphas, best y): ", best_alphas, y_best)
    return best_alphas


def DPP_MAX(GP_s, alphas, batch_size, cheap_set, prev_xs, prev_ys, ground_truth_y, ref_point):
    dim = len(cheap_set[0])
    m_kernel = mixed_kernel(GP_s, alphas)
    mixed_GPR = singleGP(dim=dim, main_kernel=m_kernel, cur_GP=None)
    for ii in range(len(prev_xs)):
        mixed_GPR.AddPointsToGP(prev_xs[ii], prev_ys[ii])

    batch = []

    for i in range(batch_size):

        m, vs = mixed_GPR.evaluate(cheap_set, std=True)
        sorted_indices = np.argsort(vs)
        c = len(sorted_indices) - 1

        while c > 0 and (sorted_indices[c] in batch or cheap_set[sorted_indices[c]] in prev_xs):
            c -= 1

        new_batch_ind = sorted_indices[c]
        batch.append(new_batch_ind)

        predictions = []
        for j in GP_s:
            mean = j.gp.predict(np.asarray([cheap_set[new_batch_ind]]), return_std=False)
            predictions.append(mean[0])
        cur_hv = calculate_contribution_HV(ground_truth_y, [predictions], ref_point)
        mixed_GPR = singleGP(dim=dim, main_kernel=m_kernel, cur_GP=None)

        for ii in range(len(prev_xs)):
            mixed_GPR.AddPointsToGP(prev_xs[ii], prev_ys[ii])
        mixed_GPR.AddPointsToGP(cheap_set[new_batch_ind], cur_hv)

    return batch


def DPP(GP_s, batch_size, alphas, cheap_pareto_set, ground_truth_x, hvs, ground_truth_y, ref_point):
    ans = DPP_MAX(GP_s, alphas, batch_size, cheap_pareto_set, ground_truth_x, hvs, ground_truth_y, ref_point)
    return ans[:batch_size]
