"""Isolated review probes; loads actual definitions without optional imports.

This is not an integration test. Run with a Python containing numpy and pandas.
"""
import ast
import copy
from pathlib import Path
from typing import Dict, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

ns = dict(np=np, pd=pd, copy=copy, Dict=Dict, Union=Union, ABC=ABC,
          abstractmethod=abstractmethod)

def load(path, names=None):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)

exec(Path('tm/constants.py').read_text(), ns)
load('tm/containers.py')
load('tm/base/base.py')
load('tm/base/lr.py', ['LinRegr'])
load('tm/base/gaussian.py', ['uGaussian'])
load('tm/base/model_converters.py')
load('tm/allocation/allocation.py')
load('tm/post_process.py', ['prepare_portfolio', 'analyze_portfolio'])
Data = ns['Data']

def probe(name, fn):
    try:
        print(name, '=>', fn())
    except Exception as exc:
        print(name, '=>', type(exc).__name__, str(exc))

probe('LinRegr small/unfitted prediction', lambda: ns['LinRegr']().posterior_predictive(x=np.ones((5, 1))))

def order():
    d = Data(ts=np.arange(3), y=np.ones((3, 1)), y_cols=['y1'],
             x=np.tile([20., 10.], (3, 1)), x_cols=['x2', 'x1'])
    out = d._get_columns(['y1', 'x1', 'x2'])
    return {'labels': out.x_cols, 'values': out.x[0].tolist()}
probe('Feature ordering', order)

def refit():
    m = ns['AsUnivariate'](ns['uGaussian']())
    m.estimate(y=np.ones((20, 1)))
    m.estimate(y=np.full((20, 1), 9.))
    return {'stored_models': len(m.base_models), 'first_model_mean': m.base_models[0].m}
probe('Repeated adapter fit', refit)
probe('max_w=0.1', lambda: ns['Optimal'](max_w=.1).get_weight(np.array([[2.]]), np.array([[[1.]]])).tolist())

def portfolio(sw, returns, fee=0., resample=None, seq=False):
    d = Data(ts=np.array([0, 3600]), y=np.zeros((2, 1)), y_cols=['y1'],
             s=np.array(returns), sw=np.array(sw), w=np.ones((2, 1)))
    p = ns['prepare_portfolio']([{'a': d}], weight_attr='sw', resample_to=resample, seq_fees=seq)
    return ns['analyze_portfolio'](p, pct_fee=fee, view=False)['strategy'].ravel().tolist()
probe('Resample returns [0.1,0.2], allocations [1,0]', lambda: portfolio([1., 0.], [.1, .2], resample='D'))
probe('Negative allocation and positive fees', lambda: portfolio([-1., -1.], [0., 0.], fee=.01))
probe('Allocation changes, constant asset weight, sequential fees', lambda: portfolio([0., 1.], [0., 0.], fee=.01, seq=True))

files = list(Path('tm').rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
print('Syntax parsed:', len(files), 'Python modules')
for path in Path('.').rglob('*.ipynb'):
    if '.ipynb_checkpoints' in str(path):
        continue
    import json
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = [''.join(c.get('source', [])) for c in notebook.get('cells', []) if c.get('cell_type') == 'code']
    print('Notebook:', path, '| code cells:', len(sources), '| external file references:',
          sum(any(s in source for s in ['read_csv', 'read_pickle', 'read_parquet', 'load_model']) for source in sources))
