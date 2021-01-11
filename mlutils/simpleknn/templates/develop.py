import warnings
import pickle

from mlutils.simpleknn.templates import BaseTemplate
from mlutils.simpleknn import TemplateSelector

from mlutils.metrics.pairwise import euclidean_distances, angular_distances


@TemplateSelector.register('develop')
class Develop(BaseTemplate):
    METRICS = ('angular', 'ang', 'cosine', 'cos', 'euclidean', 'euc')

    @classmethod
    def isAvailable(cls):
        return True

    def onInit(self, *args, **kwargs):
        warnings.warn('''
        Using Develop strategy since better could not be found.
        Develop strategy can be very slow and should not be used
        for high dimensional data and not to be used in production.

        Consider using efficient algorithm like Annoy or by sklearn.
        ''')
        metric = kwargs['metric']
        assert metric in self.__class__.METRICS, \
            'Develop can only use angular or euclidean distance fns'
        if metric in ('angular', 'cosine', 'ang', 'cos'):
            self.distance_fn = angular_distances
        elif metric == 'euclidean' or metric == 'euc':
            self.distance_fn = euclidean_distances
        self.vectors = dict()

    def onFit(self, *args, **kwargs):
        idx = self.names[kwargs['name']]
        self.vectors[idx] = tuple(float(d) for d in kwargs['vector'])

    def onBuild(self, *args, **kwargs):
        pass

    def onQuery(self, query, n):
        distances = []
        for idx, vector in self.vectors.items():
            if query == vector:
                continue
            distance = self.distance_fn(vector, query)
            distances.append((idx, vector, distance))
        distances.sort(key=lambda tup: tup[2])
        _range = n if n <= len(distances) else len(distances)
        for i in range(_range):
            yield distances[i]

    def getDistance(self, idx1, idx2):
        vec1, vec2 = self.vectors[idx1], self.vectors[idx2]
        return self.distance_fn(vec1, vec2)

    def getVector(self, idx):
        return self.vectors[idx]

    def onSave(self, file_name):
        with open(f'{file_name}-develop.pkl', 'wb') as f:
            pickle.dump(self, f)

    def onLoad(self, file_name):
        with open(f'{file_name}-develop.pkl', 'rb') as f:
            obj = pickle.load(f)
        self.vectors = obj.vectors
        self.distance_fn = obj.distance_fn
