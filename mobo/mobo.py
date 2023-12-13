import numpy as np
from .factory import init_from_config
from .transformation import StandardTransform
from .surrogate_problem import SurrogateProblem
from .utils import Timer, find_pareto_front, calc_hypervolume


class MOBO:
    config = {}

    def __init__(self, problem, n_iter, ref_point, framework_args):
        
        '''
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        '''
        self.real_problem = problem
        self.n_var, self.n_obj = problem.n_var, problem.n_obj
        self.n_iter = n_iter
        self.ref_point = ref_point

        bounds = np.array([problem.xl, problem.xu])
        self.transformation = StandardTransform(bounds)  # data normalization for surrogate model fitting

        # framework components
        framework_args['surrogate']['n_var'] = self.n_var  # for surrogate fitting
        framework_args['surrogate']['n_obj'] = self.n_obj  # for surrogate fitting
        framework_args['solver']['n_obj'] = self.n_obj
        framework = init_from_config(self.config, framework_args)

        self.surrogate_model = framework['surrogate']
        self.acquisition = framework['acquisition']
        self.solver = framework['solver']  # multi-objective solver for finding the paretofront
        self.selection = framework['selection']

        # to keep track of data and pareto information (current status of algorithm)
        self.X = None
        self.Y = None
        self.sample_num = 0
        self.status = {
            'pset': None,
            'pfront': None,
            'hv': None,
            'ref_point': self.ref_point,
        }

        # other component-specific information that needs to be stored or exported
        self.info = None

    def _update_status(self, X, Y):
        '''
        Update the status of algorithm from data
        '''
        if self.sample_num == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
        self.sample_num += len(X)

        self.status['pfront'], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status['pset'] = self.X[pfront_idx]
        self.status['hv'] = calc_hypervolume(self.status['pfront'], self.ref_point)

    def solve(self, X_init, Y_init):
        '''
        Solve the real multi-objective problem from initial data (X_init, Y_init)
        '''
        # determine reference point from data if not specified by arguments
        if self.ref_point is None:
            self.ref_point = np.max(Y_init, axis=0)
        self.selection.set_ref_point(self.ref_point)
        self.acquisition.set_ref_point(self.ref_point)
        self._update_status(X_init, Y_init)

        global_timer = Timer()

        for i in range(self.n_iter):
            print('========== Iteration %d ==========' % i)

            timer = Timer()

            # data normalization
            self.transformation.fit(self.X, self.Y)
            X, Y = self.transformation.do(self.X, self.Y)

            # build surrogate models
            self.surrogate_model.fit(X, Y)
            timer.log('Surrogate model fitted')

            self.acquisition.set_args(self.selection.batch_size, self.real_problem, self.solver,
                                      self.surrogate_model, self.transformation)
            self.acquisition.fit(X, Y)
            res = self.acquisition.final_batch()
            X_next = np.asarray([np.asarray(q) for q in res])

            # update dataset
            Y_next = self.real_problem.evaluate(X_next)
            self._update_status(X_next, Y_next)

            # statistics
            global_timer.log('Total runtime', reset=False)
            print('Total evaluations: %d, hypervolume: %.4f\n' % (self.sample_num, self.status['hv']))

            # return new data iteration by iteration
            yield X_next, Y_next

    def __str__(self):
        return \
            '========== Framework Description ==========\n' + \
            f'# algorithm: {self.__class__.__name__}\n' + \
            f'# surrogate: {self.surrogate_model.__class__.__name__}\n' + \
            f'# acquisition: {self.acquisition.__class__.__name__}\n' + \
            f'# solver: {self.solver.__class__.__name__}\n' + \
            f'# selection: {self.selection.__class__.__name__}\n'
