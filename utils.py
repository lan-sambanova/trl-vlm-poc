import os

import torch


def get_current_device() -> torch.device:
    """Gets the current available device"""
    if torch.cuda.is_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)
