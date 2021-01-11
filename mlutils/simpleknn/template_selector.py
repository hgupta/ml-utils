import inspect
import warnings


class TemplateSelector(object):
    templates = {}

    @classmethod
    def select(cls, name=None):
        def _select(name):
            if name is None:
                return None
            if cls.templates[name].isAvailable():
                return cls.templates[name]
            return None

        return _select(name) or \
            _select('develop')

    @classmethod
    def register(cls, name):
        def wrapper(klass):
            assert name not in cls.templates, \
                f'Template `{name}` alredy registered'

            cls.templates[name] = klass
            return klass
        return wrapper

