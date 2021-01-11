import pickle

from abc import abstractmethod
from collections import defaultdict


class BaseTemplate(object):
    @classmethod
    def isAvailable(cls):
        return False

    def __init__(self, *args, **kwargs):
        assert 'dims' in kwargs, 'Argument `dims` missing'
        assert 'metric' in kwargs, 'Argument `metric` missing'
        self.dims = kwargs['dims']
        self.metric = kwargs['metric']
        self.names = defaultdict(int)
        self.names_idx_map = None
        self._built = False
        self.onInit(*args, **kwargs)

    @abstractmethod
    def onInit(self, *args, **kwargs):
        pass

    def __len__(self):
        return len(self.names.keys())

    @property
    def built(self):
        return self._built

    @built.setter
    def built(self, v):
        assert self._built is False, 'Index already built, cannot rebuild'
        self._built = True

    def fit(self, *args, **kwargs):
        def fitSingle(**kw):
            if 'name' not in kw and 'vector' not in kw:
                return False
            name = kw['name']
            assert name not in self.names, \
                f'Duplicate name `{name}` encountered'
            self.names[name] = len(self)
            self.onFit(*args, **{**kwargs, **kw})
            return True

        def fitMany():
            if 'vectors' not in kwargs:
                return False
            vectors = kwargs['vectors']
            assert isinstance(vectors, dict), 'Vectors must be a dictionary'
            for name, vector in vectors.items():
                fitSingle(name=name, vector=vector)
            return True

        assert self.built is False, \
            'Index already built, can\'t insert new items'

        fitSingle(**kwargs) or fitMany()
        if 'build' in kwargs and kwargs['build']:
            self.build(*args, **kwargs)

    @abstractmethod
    def onFit(self, *args, **kwargs):
        pass

    def build(self, *args, **kwargs):
        assert self.built is False, \
            'Index already built, can\'t insert new items'
        self.onBuild(*args, **kwargs)
        self.names_idx_map = {idx: name for name, idx in self.names.items()}
        self.built = True

    @abstractmethod
    def onBuild(self, *args, **kwargs):
        pass

    def query(self, q, n=10):
        assert self.built, 'Index not built'

        _q = q
        if isinstance(q, str) and q in self.names:
            _q = self.vector(q)

        for idx, vector, distance in self.onQuery(_q, n):
            yield (self.names_idx_map[idx], vector, distance)

    @abstractmethod
    def onQuery(self, query, n):
        pass

    def distance(self, name1, name2):
        assert name1 in self.names, f'Vector `{name1}` missing'
        assert name2 in self.names, f'Vector `{name2}` missing'
        idx1, idx2 = self.names[name1], self.names[name2]
        return self.getDistance(idx1, idx2)

    @abstractmethod
    def getDistance(self, idx1, idx2):
        pass

    def vector(self, name):
        assert name in self.names, f'Vector with `{name}` not found'
        return self.getVector(self.names[name])

    @abstractmethod
    def getVector(self, idx):
        pass

    def save(self, file_name):
        with open(f'{file_name}-data.pkl', 'wb') as f:
            pickle.dump({
                'dims': self.dims,
                'metric': self.metric,
                'built': self.built,
                'names': self.names,
                'names_idx_map': self.names_idx_map
            }, f)
        self.onSave(file_name)

    @abstractmethod
    def onSave(self, file_name):
        pass

    def load(self, file_name):
        with open(f'{file_name}-data.pkl', 'rb') as f:
            obj = pickle.load(f)

        self.dims = obj['dims']
        self.metric = obj['metric']
        self.built = obj['built']
        self.names = obj['names']
        self.names_idx_map = obj['names_idx_map']
        self.onLoad(file_name)

    @abstractmethod
    def onLoad(self, file_name):
        pass

