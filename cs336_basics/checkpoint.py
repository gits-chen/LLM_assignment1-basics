from os import PathLike
from typing import IO

import torch

from cs336_basics.utils import capture_rng_state, restore_rng_state


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | PathLike[str] | IO[bytes],
) -> None:
    """保存模型、优化器、随机数状态和已完成的训练步数。

    :param model: 需要保存的模型。
    :param optimizer: 需要保存的优化器。
    :param iteration: 已完成的训练步数。
    :param out: 保存路径或可写的二进制文件对象。
    :return: 无返回值。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "rng_state": capture_rng_state(),
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | PathLike[str] | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """恢复模型、优化器和随机数状态，兼容没有随机数状态的旧检查点。

    :param src: 检查点路径或可读的二进制文件对象。
    :param model: 接收模型状态的模型。
    :param optimizer: 接收优化器状态的优化器。
    :return: 检查点记录的已完成训练步数。
    """
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

    return checkpoint["iteration"]
