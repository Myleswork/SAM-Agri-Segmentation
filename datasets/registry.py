"""Dataset registry; adapters register their own implementations."""
_REGISTRY = {}

def register(name):
    def decorate(cls):
        if name in _REGISTRY: raise ValueError(f'duplicate dataset: {name}')
        _REGISTRY[name] = cls
        return cls
    return decorate

def build_dataset(dataset_name, split, config):
    from datasets.adapters import weedyrice  # register built-in adapters
    if dataset_name not in _REGISTRY: raise ValueError(f'unknown dataset: {dataset_name}')
    return _REGISTRY[dataset_name](split, config)
