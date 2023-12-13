
def get_surrogate_model(name):
    from .surrogate_model import GaussianProcess

    surrogate_model = {
        'gp': GaussianProcess,
    }
    surrogate_model['default'] = GaussianProcess

    return surrogate_model[name]


def get_acquisition(name):
    from .acquisition import AcquisitionAdaptiveHedge

    acquisition = {
        'adaptive-hedge': AcquisitionAdaptiveHedge,
    }
    acquisition['default'] = AcquisitionAdaptiveHedge

    return acquisition[name]


def get_solver(name):
    from .solver import NSGA2Solver

    solver = {
        'nsga2': NSGA2Solver,
    }
    solver['default'] = NSGA2Solver

    return solver[name]


def get_selection(name):
    from .selection import DPPSelect

    selection = {
        'dpp': DPPSelect,
    }
    selection['default'] = DPPSelect

    return selection[name]


def init_from_config(config, framework_args):
    '''
    Initialize each component of the MOBO framework from config
    '''
    init_func = {
        'surrogate': get_surrogate_model,
        'acquisition': get_acquisition,
        'selection': get_selection,
        'solver': get_solver,
    }

    framework = {}

    for key, func in init_func.items():

        kwargs = framework_args[key]
        if config is None:
            name = kwargs[key]
        else:
            name = config[key] if key in config else 'default'

        framework[key] = func(name)(**kwargs)

    return framework
