import logging
import os
import sys

import torch
from packaging import version


_logger = logging.getLogger(__name__)


def load(name):
    if not name.startswith("dinov3_"):
        raise ValueError(
            f"Unsupported encoder: {name}. This submission only supports DINOv3 encoders."
        )
    return load_dinov3(name)


def load_dinov3(name):
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "DINOv3 requires Python 3.10+ because the official repository uses "
            "modern type-hint syntax such as `float | None`. "
            f"Current Python: {sys.version.split()[0]}. "
            "Create/use a Python 3.10+ environment, then rerun with the same DINOV3_WEIGHTS path."
        )
    if version.parse(torch.__version__.split("+")[0]) < version.parse("2.7.1"):
        raise RuntimeError(
            "DINOv3 official code expects PyTorch >= 2.7.1. "
            f"Current torch: {torch.__version__}. "
            "Upgrade torch/torchvision in this Python 3.10 environment before loading DINOv3."
        )

    repo_dir = os.environ.get("DINOV3_REPO", "./dinov3")
    weights = os.environ.get("DINOV3_WEIGHTS", None)
    repo_dir = os.path.abspath(repo_dir)

    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(
            "DINOv3 encoder requested, but the DINOv3 repository was not found. "
            f"Set DINOV3_REPO to a local facebookresearch/dinov3 checkout. Current: {repo_dir}"
        )

    try:
        try:
            import torch._dynamo  # noqa: F401
        except ImportError as dynamo_exc:
            raise RuntimeError(
                "DINOv3 requires a PyTorch build with torch._dynamo support. "
                f"Current torch version: {torch.__version__}. "
                "Install a newer PyTorch version in the Python 3.10 environment."
            ) from dynamo_exc

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from dinov3.hub import backbones as dinov3_backbones

        model_builder = getattr(dinov3_backbones, name)
        model_kwargs = {"pretrained": True}
        if weights:
            model_kwargs["weights"] = weights
        model = model_builder(**model_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the DINOv3 backbone. Check DINOV3_REPO, DINOV3_WEIGHTS, "
            "and the installed torch version."
        ) from exc

    model.is_dinov3 = True
    model.num_register_tokens = getattr(model, "n_storage_tokens", 0)
    return model
