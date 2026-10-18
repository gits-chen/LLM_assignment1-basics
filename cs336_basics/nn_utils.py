import torch

from collections.abc import Iterable


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算数值稳定的平均交叉熵损失。

    :param logits: 形状为 (..., vocab_size) 的预测分数。
    :param targets: 形状为 (...) 的正确 token ID，类型为 torch.int64。
    :return: 对所有预测位置取平均后的标量损失。
    """
    shifted_logits = (
        logits - logits.max(dim=-1, keepdim=True).values
    )

    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )

    target_logits = shifted_logits.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - target_logits

    return losses.mean()

@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """当整体梯度的 L2 范数超过限制时，等比例缩小梯度。

    :param parameters: 模型参数的可迭代对象。
    :param max_l2_norm: 允许的最大梯度 L2 范数，不能为负。
    :return: 无返回值，直接修改参数的梯度。
    """
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm 不能为负")

    grads: list[torch.Tensor] = []

    for parameter in parameters:
        grad = parameter.grad
        if grad is not None:
            grads.append(grad)

    if not grads:
        return

    squared_norms = [
        grad.float().square().sum()
        for grad in grads
    ]
    total_norm = torch.stack(squared_norms).sum().sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for grad in grads:
            grad.mul_(scale)