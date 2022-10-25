from wsipipe.load.datasets.loader import Loader
from wsipipe.load.datasets.camelyon16 import Camelyon16Loader
from wsipipe.load.datasets.stripai import StripaiLoader


loaders = {}

def register_loader(loader_cls) -> None:
    """  Add loader to loader array for loader extensions

        Args:
        loader_cls: the loader class to add
    """
    loader = loader_cls()
    loaders[loader.name] = loader


def get_loader(name: str) -> Loader:
    """A convenience function to call loader based on its name

    Args:
        name (str): Name of the loader.
    Returns:
        loader (Loader): the loader class
    """
    # print(f"calling get_loader({name})")
    if name in loaders:
        return loaders[name]
    else:
        constructor = eval(name)
        loader = constructor()
        loaders[name] = loader
        return loader