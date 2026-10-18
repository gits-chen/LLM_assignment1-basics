import math

import torch


def run_experiment(initial_lr: float) -> None:
    """观察指定初始学习率下，损失在十轮更新中的变化。

    :param initial_lr: 初始学习率。
    :return: 无返回值。
    """
    torch.manual_seed(0)

    weights = torch.nn.Parameter(
        5 * torch.randn(10, 10)
    )
    optimizer = torch.optim.SGD(
        [weights],
        lr=initial_lr,
    )

    print(f"\n初始学习率：{initial_lr:g}")

    for step in range(10):
        current_lr = initial_lr / math.sqrt(step + 1)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.zero_grad()

        loss = weights.square().mean()

        print(
            f"第 {step + 1:2d} 轮，"
            f"lr={current_lr:.6g}，"
            f"loss={loss.item():.6g}"
        )

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    for lr in [1.0, 10.0, 100.0, 1000.0]:
        run_experiment(lr)