from mlutils.simpleknn.template_selector import TemplateSelector
from mlutils.simpleknn.templates import Develop


from mlutils.version import __email__, __author__, __version__  # noqa


class SimpleKNN(object):
    """
    A simplistic KNN (K Nearest Neighbors) indexing with names as keys
    """
    def __new__(cls, *args, **kwargs):
        strategy = kwargs.get('strategy')
        return TemplateSelector.select(strategy)(*args, **kwargs)


#  class NamedKNN(object):
#      """
#      A simplistic KNN (K Nearest Neighbors) indexing with names as keys
#      """
#      def __init__(self, dims, metric='angular', strategy=None):
#          self.dims = dims
#          self.metric = metric
#          self.names = defaultdict(int)
#          self._built = False
#          if strategy is None:
#              self.strategy = StrategySelector.select()(dims, metric)
#          else:
#              if isinstance(strategy, str):
#                  self.strategy = StrategySelector.select(strategy)(dims, metric)
#              else:
#                  assert issubclass(strategy, BaseStrategy), 'Invalid Strategy'
#                  self.strategy = strategy(dims, metric)

#      def __len__(self):
#          return len(self.names.keys())

#      @property
#      def built(self):
#          return self._built

#      @built.setter
#      def built(self, v):
#          assert self._built is False, 'Index already built, cannot rebuild'
#          self._built = True

#      def distance(self, name1, name2):
#          return self.strategy.distance(self.names[name1], self.names[name2])

#      def vector(self, name):
#          return self.strategy.vector(self.names[name])

#      def insert(self, name, vector):
#          assert name not in self.names, 'Duplicate name `{name}` encountered'

#          assert self.built is False, \
#              'Index already built, can\'t insert new items'
#          self.names[name] += len(self.names.keys())
#          self.strategy.insert(self.names[name], vector)

#      def insertMany(self, items):
#          for name, vector in items:
#              self.insert(name, vector)

#      def build(self, **kwargs):
#          '''
#          builds the index

#          parameters are dependent on other strategies
#          and has different meaning for different strategies

#          develop strategy doesn't really builds an index so
#          parameters are simply ignored
#          '''
#          self.strategy.build(**kwargs)
#          self.built = True

#      def nearestByName(self, name, n=10):
#          return self.nearestByVector(self.vec(name), n)

#      def nearestByVector(self, vector, n=10):
#          return [
#              (self.names[i], vec)
#              for i, vec
#              in self.strategy.nearestByVector(vector, n)
#          ]

#      def save(self, file_name):
#          '''Saving SimpleKNN data'''
#          with open(f'{file_name}-data.pkl', 'wb', encoding='utf-8') as f:
#              pickle.dump({
#                  'dims': self.dims,
#                  'metric': self.metric,
#                  'built': self.built,
#                  'strategy': self.strategy.__class__
#              }, f)
#          '''Saving strategyic data'''
#          self.strategy.save(file_name)

#      @classmethod
#      def load(klass, file_name):
#          '''Loading SimpleKNN data'''
#          with open(f'{file_name}-data.pkl', 'rb', encoding='utf-8') as f:
#              data = pickle.load(f)

#          obj = klass(
#              dims=data.dims,
#              metric=data['metric'],
#              strategy=data['strategy']
#          )
#          obj.built = obj['built']

#          '''Loading strategyic data'''
#          obj.strategy.load(file_name)

#          return obj

