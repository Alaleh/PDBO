import os
import numpy as np
import pandas as pd
from utils import get_result_dir
from mobo.utils import find_pareto_front, calc_hypervolume

'''
Export csv files for external visualization.
'''


class DataExport:

    def __init__(self, optimizer, X, Y, args):
        self.optimizer = optimizer
        self.problem = optimizer.real_problem
        self.n_var, self.n_obj = self.problem.n_var, self.problem.n_obj
        self.batch_size = self.optimizer.selection.batch_size
        self.iter = 0
        self.transformation = optimizer.transformation

        self.result_dir = get_result_dir(args)

        n_samples = X.shape[0]

        pfront, pidx = find_pareto_front(Y, return_index=True)
        if args.ref_point is None:
            args.ref_point = np.max(Y, axis=0)
        hv_value = calc_hypervolume(pfront, ref_point=args.ref_point)

        column_names = ['iterID']
        d1 = {'iterID': np.zeros(n_samples, dtype=int)}

        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X[:, i]
            column_names.append(var_name)

        for i in range(self.n_obj):
            obj_name = f'f{i + 1}'
            d1[obj_name] = Y[:, i]


        d1['Hypervolume_indicator'] = np.full(n_samples, hv_value)
        self.export_data = pd.DataFrame(data=d1)


    def update(self, X_next, Y_next):
        '''
        For each algorithm iteration adds data for visualization.
        Input:
            X_next: proposed sample values in design space
            Y_next: proposed sample values in performance space
        '''
        self.iter += 1

        val = self.optimizer.surrogate_model.evaluate(self.transformation.do(x=X_next), std=True)
        acquisition, _, _ = self.optimizer.acquisition.evaluate(val)
        hv_value = self.optimizer.status['hv']

        d1 = {'iterID': np.full(self.batch_size, self.iter, dtype=int)}  # export all data

        for i in range(self.n_var):
            var_name = f'x{i + 1}'
            d1[var_name] = X_next[:, i]

        for i in range(self.n_obj):
            col_name = f'f{i + 1}'
            d1[col_name] = Y_next[:, i]

        d1['Hypervolume_indicator'] = np.full(self.batch_size, hv_value)

        df1 = pd.DataFrame(data=d1)
        self.export_data = self.export_data.append(df1, ignore_index=True)


    def write_csvs(self):
        '''
        Export data to csv files.
        '''
        dataframes = [self.export_data]
        filenames = ['EvaluatedSamples']

        for dataframe, filename in zip(dataframes, filenames):
            filepath = os.path.join(self.result_dir, filename + '.csv')
            dataframe.to_csv(filepath, index=False)
