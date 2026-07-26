###
# Author: Kai Li
# Date: 2022-02-12 15:16:35
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-10-04 16:24:53
###

from .hls_classifier import HLSClassifier, build_from_checkpoint, load_checkpoint

__all__ = [
   "SPMamba",
   "HLSClassifier",
   "build_from_checkpoint",
   "load_checkpoint",
]

# SPMamba được nạp lười. Nó kéo theo mamba_ssm + causal-conv1d, vốn là các gói
# biên dịch cho CUDA. Bộ phân loại tim/phổi không dùng tới SPMamba, nên nếu nạp
# sẵn ở đây thì việc huấn luyện hay chạy kiểm tra nhanh bộ phân loại sẽ đòi hỏi
# một runtime CUDA mà nó không hề cần. Truy cập ``look2hear.models.SPMamba`` hay
# ``get("SPMamba")`` vẫn hoạt động y như trước.
_LAZY_MODELS = {"SPMamba": ".SPMamba"}


def __getattr__(name):
    if name in _LAZY_MODELS:
        import importlib

        module = importlib.import_module(_LAZY_MODELS[name], __name__)
        cls = getattr(module, name)
        globals()[name] = cls          # nạp một lần, các lần sau lấy thẳng
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_model(custom_model):
    """Register a custom model, gettable with `models.get`.

    Args:
        custom_model: Custom model to register.

    """
    if (
        custom_model.__name__ in globals().keys()
        or custom_model.__name__.lower() in globals().keys()
    ):
        raise ValueError(
            f"Model {custom_model.__name__} already exists. Choose another name."
        )
    globals().update({custom_model.__name__: custom_model})


def get(identifier):
    """Returns an model class from a string (case-insensitive).

    Args:
        identifier (str): the model name.

    Returns:
        :class:`torch.nn.Module`
    """
    if isinstance(identifier, str):
        to_get = {k.lower(): v for k, v in globals().items()}
        cls = to_get.get(identifier.lower())
        if cls is None:
            # Chưa nạp thì tìm trong bảng nạp lười trước khi báo lỗi.
            for name in _LAZY_MODELS:
                if name.lower() == identifier.lower():
                    return __getattr__(name)
            raise ValueError(f"Could not interpret model name : {str(identifier)}")
        return cls
    raise ValueError(f"Could not interpret model name : {str(identifier)}")
