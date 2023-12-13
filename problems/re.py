import os
import numpy as np
from .problem import Problem
from abc import ABC, abstractmethod


def div(x1, x2):
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


class RE(Problem, ABC):
    '''
    Tanabe, Ryoji, and Hisao Ishibuchi. "An easy-to-use real-world multi-objective optimization problem suite." Applied Soft Computing (2020): 106078.
    '''
    n_var = None
    n_obj = None
    xl = None
    xu = None

    def __init__(self):
        Problem.__init__(self, n_var=self.n_var, n_obj=self.n_obj, xl=np.array(self.xl), xu=np.array(self.xu))

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        if requires_F:
            out['F'] = np.column_stack([*self._evaluate_F(x)])

    @abstractmethod
    def _evaluate_F(self, x):
        pass

    def _calc_pareto_front(self, *args, **kwargs):
        name = self.__class__.__name__
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/RE/ParetoFront/{name}.npy')
        return np.load(file_path)


class GearTrainDesign(RE):
    n_var = 4
    n_obj = 3
    xl = [12] * 4
    xu = [60] * 4

    def _evaluate_F(self, x):
        x1, x2, x3, x4 = np.round(x[:, 0]), np.round(x[:, 1]), np.round(x[:, 2]), np.round(x[:, 3])
        f1 = np.abs(6.931 - (div(x3, x1) * div(x4, x2)))
        f2 = np.max(np.column_stack([x1, x2, x3, x4]), axis=1)
        g = 0.5 - (f1 / 6.931)
        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]
        f3 = g

        return f1, f2, f3
