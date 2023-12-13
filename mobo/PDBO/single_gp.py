import copy
import numpy as np
from mobo.utils import safe_divide
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from mobo.surrogate_model.base import SurrogateModel
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import cholesky, cho_solve, solve_triangular
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


class singleGP():

    def __init__(self, dim, main_kernel=None, **kwargs):

        def constrained_optimization(obj_func, initial_theta, bounds):
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
            return opt_res.x, opt_res.fun

        self.X = []
        self.y = []
        nu = 5

        if main_kernel is None:
            if nu > 0:
                main_kernel = Matern(length_scale=np.ones(dim), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                                     nu=0.5 * nu)
            else:
                main_kernel = RBF(length_scale=np.ones(dim), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))

            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-3), np.sqrt(1e3))) * \
                     main_kernel + ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))

            self.gp = GaussianProcessRegressor(kernel=kernel, optimizer=constrained_optimization, normalize_y=True)
        else:
            self.gp = GaussianProcessRegressor(kernel=main_kernel, n_restarts_optimizer=0, random_state = 11)
            self.kernel = main_kernel

        self.gp._K_inv = None
        self.K = None

    def fit(self, X, Y):
        self.X = X
        self.y = Y
        self.gp.fit(X, Y)

    def addSample(self, x, y):

        if isinstance(x[0], list):
            self.X.extend(x)
            self.y.extend(y)
        else:
            self.X.append(x)
            self.y.append(y)

    def evaluate_sklearn(self, X, std = True):
        X = np.array(X)
        mean, stddev = self.gp.predict(X.reshape(1, -1), return_std=True)
        if std:
            return mean, stddev
        return mean[0]

    def evaluate(self, X, std=True):

        Ker = self.gp.kernel_(X, self.gp.X_train_)  
        self.mean = Ker.dot(self.gp.alpha_)

        if not std:
            return self.mean

        if self.gp._K_inv is None:
            L_inv = solve_triangular(self.gp.L_.T, np.eye(self.gp.L_.shape[0]))
            self.gp._K_inv = L_inv.dot(L_inv.T)

        y_var = self.gp.kernel_.diag(X)
        y_var -= np.einsum("ij,ij->i", np.dot(Ker, self.gp._K_inv), Ker)

        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            y_var[y_var_negative] = 0.0

        self.std = np.sqrt(y_var)

        return self.mean, self.std

    def getCovarianceMatrix(self, points):
        p = copy.deepcopy(points)
        self.K = self.gp.kernel_.__call__(np.asarray(p))
        return self.K

    def AddPointsToGP(self, new_x, new_Y):

        self.addSample(new_x, new_Y)
        self.gp.kernel_ = clone(self.gp.kernel)

        X = np.asarray(copy.deepcopy(self.X))
        y = np.asarray(copy.deepcopy(self.y))

        # Normalize target value
        if self.gp.normalize_y:
            self.gp._y_train_mean = np.mean(y, axis=0)
            self.gp._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self.gp._y_train_mean) / self.gp._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self.gp._y_train_mean = np.zeros(shape=shape_y_stats)
            self.gp._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.gp.alpha) and self.gp.alpha.shape[0] != y.shape[0]:
            if self.gp.alpha.shape[0] == 1:
                self.gp.alpha = self.gp.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.gp.alpha.shape[0]} != {y.shape[0]})"
                )

        self.gp.X_train_ = np.copy(X) if self.gp.copy_X_train else X
        self.gp.y_train_ = np.copy(y) if self.gp.copy_X_train else y

        K = self.gp.kernel_(self.gp.X_train_)
        K[np.diag_indices_from(K)] += self.gp.alpha
        GPR_CHOLESKY_LOWER = True
        try:
            self.gp.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                           f"The kernel, {self.gp.kernel_}, is not returning a positive "
                           "definite matrix. Try gradually increasing the 'alpha' "
                           "parameter of your GaussianProcessRegressor estimator.",
                       ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.gp.alpha_ = cho_solve((self.gp.L_, GPR_CHOLESKY_LOWER),self.gp.y_train_,check_finite=False,)


def clone(estimator, *, safe=True):
    """Construct a new unfitted estimator with the same parameters.
    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It returns a new estimator
    with the same parameters that has not been fitted on any data.
    Parameters
    ----------
    estimator : {list, tuple, set} of estimator instance or a single \
            estimator instance
        The estimator or group of estimators to be cloned.
    safe : bool, default=True
        If safe is False, clone will fall back to a deep copy on objects
        that are not estimators.
    Returns
    -------
    estimator : object
        The deep copy of the input, an estimator if input is an estimator.
    Notes
    -----
    If the estimator's `random_state` parameter is an integer (or if the
    estimator doesn't have a `random_state` parameter), an *exact clone* is
    returned: the clone and the original estimator will give the exact same
    results. Otherwise, *statistical clone* is returned: the clone might
    return different results from the original estimator. More details can be
    found in :ref:`randomness`.
    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError(
                    "Cannot clone object. "
                    + "You should provide an instance of "
                    + "scikit-learn estimator instead of a class."
                )
            else:
                raise TypeError(
                    "Cannot clone object '%s' (type %s): "
                    "it does not seem to be a scikit-learn "
                    "estimator as it does not implement a "
                    "'get_params' method." % (repr(estimator), type(estimator))
                )

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s" % (estimator, name)
            )
    return new_object
