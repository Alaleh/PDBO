import numpy as np
from mobo.PDBO.DPP import *
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.nsga2 import calc_crowding_distance


class Selection(ABC):
    '''
    Base class of selection method
    '''
    def __init__(self, batch_size, ref_point=None, **kwargs):
        self.batch_size = batch_size
        self.ref_point = ref_point

    def fit(self, X, Y):
        '''
        Fit the parameters of selection method from data
        '''
        pass

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    @abstractmethod
    def select(self, solution, surrogate_model, status, transformation):
        '''
        Select new samples from solution obtained by solver
        Input:
            solution['x']: design variables of solution
            solution['y']: acquisition values of solution
            solution['algo']: solver algorithm, having some relevant information from optimization
            surrogate_model: fitted surrogate model
            status['pset']: current pareto set found
            status['pfront]: current pareto front found
            status['hv']: current hypervolume
            transformation: data normalization for surrogate model fitting
            (some inputs may not be necessary for some selection criterion)
        Output:
            X_next: next batch of samples selected
            info: other informations need to be stored or exported, None if not necessary
        '''
        pass


class DPPSelect(Selection):
    '''
    Selection method for ABBO algorithm
    '''

    def __init__(self, batch_size, ref_point=None, **kwargs):
        self.batch_size = batch_size
        self.ref_point = ref_point
        self.results = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def select(self, solution, surrogate_model, status, transformation):

        if self.results is not None:
            return np.asarray(self.results), None

        X = solution['x']
        cheap_points = copy.deepcopy(solution['x'])
        ground_truth_x = list(copy.deepcopy(surrogate_model.X))
        ground_truth_y = copy.deepcopy(surrogate_model.y)

        unnormal_y = transformation.undo(y=ground_truth_y)

        cur_hv = calculate_HV(unnormal_y, self.ref_point)
        new_hv = calculate_HV(unnormal_y[1:], self.ref_point)
        diff_hvs = [cur_hv - new_hv]
        for q in range(1, len(unnormal_y) - 1):
            cur_front = np.concatenate((unnormal_y[:q], unnormal_y[q + 1:]), axis=0)
            new_hv = calculate_HV(cur_front, self.ref_point)
            diff_hvs.append(cur_hv - new_hv)
        if len(unnormal_y)>1:
            new_hv = calculate_HV(unnormal_y[:-1], self.ref_point)
            diff_hvs.append(cur_hv - new_hv)

        top_indices = find_batch(self.batch_size, cheap_points, ground_truth_x, diff_hvs, unnormal_y, self.ref_point)

        self.results = X[top_indices]
        self.results = np.asarray(self.results)
        self.results = transformation.undo(x = self.results)

        return self.results, None

