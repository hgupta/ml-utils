from math import sqrt
from mlutils.functools import lru_cache

try:
    from sklearn.metrics.pairwise import euclidean_distances
except ImportError:
    @lru_cache()
    def euclidean_distances(X, Y):
        return sqrt(sum([(i - j) ** 2 for i, j in zip(X, Y)]))


def norm(vec):
    return sqrt(sum([i ** 2 for i in vec]))


def normalize(vec):
    normalized = norm(vec)
    return tuple(i / normalized for i in vec)


@lru_cache()
def angular_distances(X, Y):
    return euclidean_distances(normalize(X), normalize(Y))


__all__ = [
    'euclidean_distances',
    'angular_distances'
]
