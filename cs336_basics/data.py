import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机抽取输入序列及其向后错开一位的目标序列。

    :param dataset: 一维 NumPy 数组，保存连续的 token ID。
    :param batch_size: 每批抽取的序列数量。
    :param context_length: 每条输入序列的长度。
    :param device: 输出张量所在设备，例如 cpu 或 cuda:0。
    :return: 两个形状为 (batch_size, context_length) 的整数张量。
    """
    if dataset.ndim != 1:
        raise ValueError("dataset 必须是一维数组")

    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size 和 context_length 必须为正")

    if len(dataset) <= context_length:
        raise ValueError("数据至少需要 context_length + 1 个 token")

    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
        dtype=np.int64,
    )

    x_batch = np.stack(
        [
            dataset[start:start + context_length]
            for start in starts
        ]
    )

    y_batch = np.stack(
        [
            dataset[start + 1:start + context_length + 1]
            for start in starts
        ]
    )

    x = torch.tensor(
        x_batch,
        dtype=torch.long,
        device=device,
    )
    y = torch.tensor(
        y_batch,
        dtype=torch.long,
        device=device,
    )

    return x, y