try:
    from functools import lru_cache
except ImportError:
    from functools import wraps

    def lru_cache(maxsize=10000):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator
