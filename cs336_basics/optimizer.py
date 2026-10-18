import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """计算带线性预热的余弦学习率。

    :param it: 当前训练步数，从 0 开始。
    :param max_learning_rate: 预热结束时达到的最大学习率。
    :param min_learning_rate: 衰减结束后的最小学习率。
    :param warmup_iters: 线性预热的步数。
    :param cosine_cycle_iters: 余弦衰减结束时的步数。
    :return: 当前训练步应使用的学习率。
    """
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError(
            "warmup_iters 不能为负，"
            "cosine_cycle_iters 必须大于 warmup_iters"
        )

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    return (
        min_learning_rate
        + 0.5
        * (1 + math.cos(math.pi * progress))
        * (max_learning_rate - min_learning_rate)
    )

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """创建 AdamW 优化器。

        :param params: 需要优化的参数或参数组。
        :param lr: 学习率。
        :param betas: 梯度及梯度平方移动平均的衰减系数。
        :param eps: 分母中用于数值稳定的小常数。
        :param weight_decay: 权重衰减系数。
        :return: 无返回值。
        """
        if lr < 0 or eps <= 0 or weight_decay < 0:
            raise ValueError("lr、weight_decay 不能为负，eps 必须为正")

        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas 中的值必须在 [0, 1) 范围内")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """根据当前梯度更新参数和优化器状态。

        :param closure: 可选的重新计算损失的函数。
        :return: closure 的返回值，未提供时返回 None。
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("此 AdamW 实现需要稠密梯度")

                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]

                p.mul_(1 - lr * weight_decay)

                state["m"] = (
                    beta1 * state["m"]
                    + (1 - beta1) * grad
                )
                state["v"] = (
                    beta2 * state["v"]
                    + (1 - beta2) * grad.square()
                )

                step_size = (
                    lr
                    * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )

                update = state["m"] / (
                    state["v"].sqrt() + eps
                )

                p.add_(update, alpha=-step_size)

        return loss