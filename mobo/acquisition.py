import math
import numpy as np
from external import lhs
from scipy.stats import norm
from numpy import linalg as LA
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from .utils import safe_divide, expand
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from scipy.stats.distributions import chi2
from mobo.PDBO.HedgeAdaptiveAcq import HedgeAdaptive
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


class Acquisition(ABC):
    '''
    Base class of acquisition function
    '''
    requires_std = False

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, Y):
        '''
        Fit the parameters of acquisition function from data
        '''
        pass

    @abstractmethod
    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        '''
        Evaluate the output from surrogate model using acquisition function
        Input:
            val: output from surrogate model, storing mean and std of prediction, and their derivatives
            val['F']: mean, shape (N, n_obj)
            val['dF']: gradient of mean, shape (N, n_obj, n_var)
            val['hF']: hessian of mean, shape (N, n_obj, n_var, n_var)
            val['S']: std, shape (N, n_obj)
            val['dS']: gradient of std, shape (N, n_obj, n_var)
            val['hS']: hessian of std, shape (N, n_obj, n_var, n_var)
        Output:
            F: acquisition value, shape (N, n_obj)
            dF: gradient of F, shape (N, n_obj, n_var)
            hF: hessian of F, shape (N, n_obj, n_var, n_var)
        '''
        pass


class IdentityFunc(Acquisition):
    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        F, dF, hF = val['F'], val['dF'], val['hF']
        return F, dF, hF


class PI(Acquisition):
    requires_std = True

    def __init__(self, *args, **kwargs):
        self.y_min = None

    def fit(self, X, Y):
        self.y_min = np.min(Y, axis=0)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        y_mean, y_std = val['F'], val['S']
        z = safe_divide(self.y_min - y_mean, y_std)
        cdf_z = norm.cdf(z)
        F = -cdf_z

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val['dF'], val['hF'], val['dS'], val['hS']

        if calc_gradient or calc_hessian:
            dz_y_mean = -safe_divide(1, y_std)
            dz_y_std = -safe_divide(self.y_min - y_mean, y_std ** 2)

            pdf_z = norm.pdf(z)
            dF_y_mean = -pdf_z * dz_y_mean
            dF_y_std = -pdf_z * dz_y_std

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            dpdf_z_z = -z * pdf_z
            dpdf_z_y_mean = dpdf_z_z * dz_y_mean
            dpdf_z_y_std = dpdf_z_z * dz_y_std
            hz_y_std = safe_divide(self.y_min - y_mean, y_std ** 3)

            hF_y_mean = -dpdf_z_y_mean * dz_y_mean
            hF_y_std = -dpdf_z_y_std * dz_y_std - pdf_z * hz_y_std

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)
            hF_y_mean, hF_y_std = expand(hF_y_mean, (-1, -2)), expand(hF_y_std, (-1, -2))

            hF = dF_y_mean * hy_mean + dF_y_std * hy_std + \
                 hF_y_mean * dy_mean * dy_mean_T + hF_y_std * dy_std * dy_std_T

        return F, dF, hF


class EI(Acquisition):
    requires_std = True

    def __init__(self, *args, **kwargs):
        self.y_min = None

    def fit(self, X, Y):
        self.y_min = np.min(Y, axis=0)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        y_mean, y_std = val['F'], val['S']
        jitter = 1e-2
        z = safe_divide(self.y_min - jitter - y_mean, y_std)
        pdf_z = norm.pdf(z)
        cdf_z = norm.cdf(z)
        F = -(self.y_min - y_mean) * cdf_z - y_std * pdf_z

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val['dF'], val['hF'], val['dS'], val['hS']

        if calc_gradient or calc_hessian:
            dz_y_mean = -safe_divide(1, y_std)
            dz_y_std = -safe_divide(self.y_min - y_mean, y_std ** 2)
            dpdf_z_z = -z * pdf_z

            dF_y_mean = cdf_z - (self.y_min - y_mean) * pdf_z * dz_y_mean - y_std * dpdf_z_z * dz_y_mean
            dF_y_std = (self.y_min - y_mean) * pdf_z * dz_y_std + pdf_z + y_std * dpdf_z_z * dz_y_std

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            dpdf_z_y_mean = dpdf_z_z * dz_y_mean
            dpdf_z_y_std = dpdf_z_z * dz_y_std
            ddpdf_z_z_y_mean = -z * dpdf_z_y_mean - dz_y_mean * pdf_z
            ddpdf_z_z_y_std = -z * dpdf_z_y_std - dz_y_std * pdf_z
            ddz_y_std_y_std = safe_divide(self.y_min - y_mean, y_std ** 3)

            hF_y_mean = -pdf_z * dz_y_mean - \
                        dz_y_mean * pdf_z + (self.y_min - y_mean) * dpdf_z_z * dz_y_mean ** 2 + \
                        y_std * dz_y_mean * ddpdf_z_z_y_mean
            hF_y_std = (self.y_min - y_mean) * (dz_y_std * dpdf_z_y_std + pdf_z * ddz_y_std_y_std) + \
                       dpdf_z_y_std + \
                       dpdf_z_z * dz_y_std + y_std * dz_y_std * ddpdf_z_z_y_std + y_std * dpdf_z_z * ddz_y_std_y_std

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)
            hF_y_mean, hF_y_std = expand(hF_y_mean, (-1, -2)), expand(hF_y_std, (-1, -2))

            hF = dF_y_mean * hy_mean + dF_y_std * hy_std + \
                 hF_y_mean * dy_mean * dy_mean_T + hF_y_std * dy_std * dy_std_T

        return F, dF, hF


class UCB(Acquisition):
    requires_std = True

    def __init__(self, *args, **kwargs):
        self.n_sample = None

    def fit(self, X, Y):
        self.d = X.shape[1]
        self.n_sample = X.shape[0]

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        mu = 0.5
        tau = 2 * math.log((pow(self.n_sample, (self.d / 2) + 2) * pow(math.pi, 2)) / (3 * 0.05))
        lamda = math.sqrt(mu * tau)

        y_mean, y_std = val['F'], val['S']
        F = y_mean - lamda * y_std

        dF, hF = None, None
        dy_mean, hy_mean, dy_std, hy_std = val['dF'], val['hF'], val['dS'], val['hS']

        if calc_gradient or calc_hessian:
            dF_y_mean = np.ones_like(y_mean)
            dF_y_std = -lamda * np.ones_like(y_std)

            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

        if calc_gradient:
            dF = dF_y_mean * dy_mean + dF_y_std * dy_std

        if calc_hessian:
            hF_y_mean = 0
            hF_y_std = 0

            dy_mean, dy_std = expand(dy_mean), expand(dy_std)
            dy_mean_T, dy_std_T = dy_mean.transpose(0, 1, 3, 2), dy_std.transpose(0, 1, 3, 2)
            dF_y_mean, dF_y_std = expand(dF_y_mean), expand(dF_y_std)

            hF = dF_y_mean * hy_mean + dF_y_std * hy_std + \
                 hF_y_mean * dy_mean * dy_mean_T + hF_y_std * dy_std * dy_std_T

        return F, dF, hF


class TS(Acquisition):
    requires_std = True

    def __init__(self, *args, **kwargs):
        self.X = None
        self.Y = None
        self.n_var = None
        self.n_obj = None
        self.n_sample = None
        self.mean_sample = 0

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_var = self.X.shape[1]
        self.n_obj = self.Y.shape[1]
        self.M = 100  # n_spectral_points??
        self.nu = 5
        self.gps = []

        def constrained_optimization(obj_func, initial_theta, bounds):
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
            return opt_res.x, opt_res.fun

        for _ in range(self.n_obj):
            if self.nu > 0:
                main_kernel = Matern(length_scale=np.ones(self.n_var),
                                     length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                                     nu=0.5 * self.nu)
            else:
                main_kernel = RBF(length_scale=np.ones(self.n_var), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))

            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-3), np.sqrt(1e3))) * \
                     main_kernel + \
                     ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))

            gp = GaussianProcessRegressor(kernel=kernel, optimizer=constrained_optimization)
            self.gps.append(gp)

        self.thetas, self.Ws, self.bs, self.sf2s = [], [], [], []
        n_sample = X.shape[0]

        for i, gp in enumerate(self.gps):
            gp.fit(X, Y[:, i])

            ell = np.exp(gp.kernel_.theta[1:-1])
            sf2 = np.exp(2 * gp.kernel_.theta[0])
            sn2 = np.exp(2 * gp.kernel_.theta[-1])

            sw1, sw2 = lhs(self.n_var, self.M), lhs(self.n_var, self.M)
            if self.nu > 0:
                W = np.tile(1. / ell, (self.M, 1)) * norm.ppf(sw1) * np.sqrt(self.nu / chi2.ppf(sw2, df=self.nu))
            else:
                W = np.random.uniform(size=(self.M, self.n_var)) * np.tile(1. / ell, (self.M, 1))
            b = 2 * np.pi * lhs(1, self.M)
            phi = np.sqrt(2. * sf2 / self.M) * np.cos(W @ X.T + np.tile(b, (1, n_sample)))
            A = phi @ phi.T + sn2 * np.eye(self.M)
            invcholA = LA.inv(LA.cholesky(A))
            invA = invcholA.T @ invcholA
            mu_theta = invA @ phi @ Y[:, i]
            if self.mean_sample:
                theta = mu_theta
            else:
                cov_theta = sn2 * invA
                cov_theta = 0.5 * (cov_theta + cov_theta.T)
                theta = mu_theta + LA.cholesky(cov_theta) @ np.random.standard_normal(self.M)

            self.thetas.append(theta.copy())
            self.Ws.append(W.copy())
            self.bs.append(b.copy())
            self.sf2s.append(sf2)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        X = val['X']
        F, dF, hF = [], [], []
        n_sample = max(X.shape[0], 1)

        for theta, W, b, sf2 in zip(self.thetas, self.Ws, self.bs, self.sf2s):
            factor = np.sqrt(2. * sf2 / self.M)
            W_X_b = W @ X.T + np.tile(b, (1, n_sample))
            F.append(factor * (theta @ np.cos(W_X_b)))

            if calc_gradient:
                dF.append(-factor * np.expand_dims(theta, 0) * np.sin(W_X_b).T @ W)

            if calc_hessian:
                hF.append(-factor * np.einsum('ij,jk->ikj', np.expand_dims(theta, 0) * np.cos(W_X_b).T, W) @ W)

        F = np.stack(F, axis=1)
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None

        return F, dF, hF


class AcquisitionAdaptiveHedge(Acquisition):
    requires_std = False

    def __init__(self, *args, **kwargs):
        self.acq_function = HedgeAdaptive()
        self.prob = None
        self.solver = None
        self.models = None
        self.trans = None
        self.ref_point = None
        self.batch_size = None
        self.requires_std = False
        self.X = None
        self.Y = None

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    def set_args(self, batch_size, real_problem, solver, surrogate_model, transformation):
        self.batch_size = batch_size
        self.prob = real_problem
        self.solver = solver
        self.models = surrogate_model
        self.trans = transformation

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.acq_function.x = X
        self.acq_function.y = Y
        if self.acq_function.iteration_number > 0:
            self.acq_function.update_reward()
            self.acq_function.update_probabilities()
        self.acq_function.iterate(self.X, self.Y, self.prob, self.solver, self.models, self.trans, self.batch_size,
                                  self.ref_point)

    def evaluate(self, val, calc_gradient=False, calc_hessian=False):
        possible_acqs = ["TS", "UCB", "EI", "identity"]
        self.acquisition_evaluator = get_acquisition_function(possible_acqs[self.acq_function.selected_acq],
                                                              len(self.X[0]), len(self.Y[0]))
        self.acquisition_evaluator.fit(self.X, self.Y)
        F, dF, hF = self.acquisition_evaluator.evaluate(val, calc_gradient=False, calc_hessian=False)
        return F, dF, hF

    def final_batch(self):
        res = self.acq_function.get_batch()
        res = self.trans.undo(x=res)
        return res


def get_acquisition_function(s, n_v, n_o):
    if s == "EI":
        ac = EI()
    elif s == "identity":
        ac = IdentityFunc()
    elif s == "UCB":
        ac = UCB()
    elif s == "TS":
        ac = TS(n_v, n_o)
    else:
        raise "Unimplemented acquisition function"
    return ac
