import copy
import random
import numpy as np
from numpy.random import choice
from mobo.PDBO.DPP import find_batch
from mobo.PDBO.single_gp import singleGP
from mobo.surrogate_problem import SurrogateProblem
from mobo.utils import calc_hypervolume, calculate_HV, calculate_contribution_HV

def get_acquisition_function(s, n_v, n_o):
    from mobo.acquisition import EI, UCB, TS, IdentityFunc
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


def normalize(value):
    return (value - value.mean()) / value.std()


class HedgeAdaptive:

    def __init__(self):

        random.seed(1)
        np.random.seed(1)

        self.acquisitions = ["TS", "UCB", "EI", "identity"]
        self.reference_point = None
        self.GPs = None
        self.portfolio_size = len(self.acquisitions)
        self.initial_data_size = None
        self.dim = None
        self.M = None
        self.x = None
        self.y = None
        self.disc_rewards = [0 for i in range(self.portfolio_size)]
        self.norm_rewards = [0 for i in range(self.portfolio_size)]
        self.norm_batch = []
        self.batches = []
        self.probabilities = [1.0 / self.portfolio_size for i in range(self.portfolio_size)]
        self.ch_solver = None
        self.batch_size = None
        self.iteration_number = 0
        self.eta = 4.0  
        self.gamma = 0.75

    def iterate(self, init_x, init_y, prob, solver, models, trans, batch_size, reference_point):
        self.initial_data_size = len(init_x)
        self.dim = len(init_x[0])
        self.M = len(init_y[0])
        self.x = init_x
        self.y = init_y
        self.GPs = models
        self.ch_solver = solver
        self.batch_size = batch_size
        self.batches = []
        self.transform = trans
        self.reference_point = reference_point

        self.chooseHedge()

        unnormal_y = self.transform.undo(y=self.y)
        cur_hv = calculate_HV(unnormal_y, self.reference_point)
        new_hv = calculate_HV(unnormal_y[1:], self.reference_point)
        diff_hvs = [cur_hv - new_hv]
        for q in range(1, len(unnormal_y) - 1):
            cur_front = np.concatenate((unnormal_y[:q], unnormal_y[q + 1:]), axis=0)
            new_hv = calculate_HV(cur_front, self.reference_point)
            diff_hvs.append(cur_hv - new_hv)
        if len(unnormal_y)>1:
            new_hv = calculate_HV(unnormal_y[:-1], self.reference_point)
            diff_hvs.append(cur_hv - new_hv)

        alphas = None

        from mobo.acquisition import EI, UCB, TS, IdentityFunc
        for ss in range(self.portfolio_size):
            acq = get_acquisition_function(self.acquisitions[ss], self.dim, self.M)
            acq.fit(X=self.x, Y=self.y)
            surr_problem = SurrogateProblem(prob, self.GPs, acq, self.transform)
            solution = self.ch_solver.solve(surr_problem, self.x, self.y)
            cheaps = solution['x']
            batch_indices, alphas = find_batch(self.batch_size, cheaps, self.x, diff_hvs, unnormal_y, self.reference_point, alphas, True)
            batch = [cheaps[i] for i in batch_indices[:self.batch_size]]
            self.batches.append(batch)

        self.iteration_number += 1

    def update_reward(self):

        this_batch = copy.deepcopy(self.y[:-self.batch_size])
        prev_y = np.asarray(this_batch)
        prev_y = self.transform.undo(y=prev_y)

        new_GPs = []
        for qq in range(self.M):
            new_gp = singleGP(dim=self.dim)
            new_gp.fit(self.x[:-self.batch_size], [pp[qq] for pp in prev_y])
            new_GPs.append(new_gp)

        self.norm_batch = [list(dd) for dd in self.batches]
        ests = [[new_GPs[ww].evaluate_sklearn(self.norm_batch[qq][ll], std=False) for ww in range(self.M)] for ll in
                range(self.batch_size) for qq in range(self.portfolio_size)]

        new_rewards = [0 for qq in range(self.portfolio_size)]
        hv_prev = calculate_HV(prev_y, self.reference_point)

        for qq in range(self.portfolio_size):
            cur_means = np.append(np.asarray([ests[qq]]), prev_y, axis=0)
            new_hv = calculate_HV(cur_means, self.reference_point)
            hv_contribution = new_hv - hv_prev
            new_rewards[qq] = hv_contribution
        if hv_prev != 0:
            new_rewards = [dd / hv_prev for dd in new_rewards]
        if max(self.disc_rewards) == 0 and max(new_rewards) == 0:
            portfolio_hvs = [calculate_HV(np.asarray([qq]), self.reference_point) for qq in ests]
            if hv_prev != 0:
                new_rewards = [qq / hv_prev for qq in portfolio_hvs]
            elif max(portfolio_hvs) != 0:
                new_rewards = [qq / max(portfolio_hvs) for qq in portfolio_hvs]
        self.disc_rewards = [self.gamma * self.disc_rewards[i] + new_rewards[i] for i in range(self.portfolio_size)]
        if max(self.disc_rewards) != min(self.disc_rewards):
            self.norm_rewards = [
                (self.disc_rewards[i] - max(self.disc_rewards)) / (max(self.disc_rewards) - min(self.disc_rewards)) for
                i in range(self.portfolio_size)]
        else:
            self.norm_rewards = [self.norm_rewards[0] for tt in range(self.portfolio_size)]

    def update_probabilities(self):
        self.probabilities = [np.exp(self.eta * (self.norm_rewards[i])) for i in range(self.portfolio_size)]
        s = sum(self.probabilities)
        self.probabilities = [i / s for i in self.probabilities]

    def chooseHedge(self):
        self.selected_acq = choice([i for i in range(len(self.probabilities))], p=self.probabilities)
        print("Selected acquisition: ", self.acquisitions[self.selected_acq])

    def get_batch(self):
        return [x.tolist() for x in self.batches[self.selected_acq]]
