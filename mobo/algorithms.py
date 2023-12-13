from .mobo import MOBO


class PDBO(MOBO):
    config = {
        'surrogate': 'gp',
        'acquisition': 'adaptive-hedge',
        'solver': 'nsga2',
        'selection': 'dpp',
    }


def get_algorithm(name):
    '''
    Get class of algorithm by name
    '''
    algo = {
        'PDBO': PDBO,
    }
    return algo[name]
