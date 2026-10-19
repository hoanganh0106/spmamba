###
# Author: Kai Li
# Date: 2022-02-12 15:16:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:24:53
###

__all__ = [
    "SPMamba",
]

_LAZY_MODELS = {"SPMamba": ".SPMamba"}


def __getattr__(name):
    if name in _LAZY_MODELS:
        import importlib

        module = importlib.import_module(_LAZY_MODELS[name], __name__)
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`."""
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns a model class from a string (case-insensitive)."""
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            for name in _LAZY_MODELS:
                if name.lower() == identifier.lower():
                    return __getattr__(name)
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
