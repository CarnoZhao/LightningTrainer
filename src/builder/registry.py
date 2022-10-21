from collections import defaultdict


REGISTRY = defaultdict(dict)

def register(name):
    def register_(cls):
        REGISTRY[name][cls.__name__] = cls
        return cls
    return register_