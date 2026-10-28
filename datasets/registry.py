"""Dataset adapters register with @register; built-ins are discovered from adapters/."""
import importlib
import pkgutil
_REGISTRY = {}
_DISCOVERED = False

def register(name):
    def decorate(cls):
        if name in _REGISTRY: raise ValueError(f'duplicate dataset: {name}')
        _REGISTRY[name] = cls
        return cls
    return decorate

def build_dataset(dataset_name, split, config):
    global _DISCOVERED
    if not _DISCOVERED:
        package=importlib.import_module('datasets.adapters')
        for mod in pkgutil.iter_modules(package.__path__): importlib.import_module(f'datasets.adapters.{mod.name}')
        _DISCOVERED=True
    if dataset_name not in _REGISTRY: raise ValueError(f'unknown dataset: {dataset_name}')
    return _REGISTRY[dataset_name](split, config)
