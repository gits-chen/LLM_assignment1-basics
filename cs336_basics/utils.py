import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy


def capture_rng_state() -> dict[str, Any]:
    """记录 NumPy、PyTorch CPU 和 CUDA 的随机数状态。

    :return: 仅包含张量和基础类型、可随检查点保存的状态字典。
    """
    numpy_state = np.random.get_state(legacy=True)

    if not isinstance(numpy_state, tuple):
        raise TypeError("预期 NumPy 随机数状态为 MT19937 元组")

    return {
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """恢复检查点中的 NumPy、PyTorch CPU 和 CUDA 随机数状态。

    :param state: capture_rng_state 返回的状态字典。
    :return: 无返回值。
    """
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )

    torch.set_rng_state(state["torch_cpu"].cpu())

    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(
            [cuda_state.cpu() for cuda_state in state["torch_cuda"]]
        )


class BoldYellowHelpFormatter(argparse.HelpFormatter):
    """以黄色粗体显示参数帮助说明。"""

    def _split_lines(self, text: str, width: int) -> list[str]:
        """先按原规则换行，再为每行帮助说明添加颜色。

        :param text: 已展开默认值的帮助说明。
        :param width: 每行帮助说明的可用宽度。
        :return: 带黄色粗体和样式重置标记的文本行。
        """
        lines = super()._split_lines(text, width)
        return [f"\033[1;33m{line}\033[0m" for line in lines]


def parse_args() -> argparse.Namespace:
    """解析并检查训练脚本的命令行参数。

    :return: 包含训练配置的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="训练 Transformer 语言模型",
        formatter_class=BoldYellowHelpFormatter,
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=20,
        help="累计训练目标步数，必须大于当前的 5 步预热（默认：%(default)s）",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="要恢复的检查点路径（默认：%(default)s，表示从头训练）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/debug_last.pt"),
        help="训练结束后的检查点保存路径（默认：%(default)s）",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="余弦衰减结束步数（默认：%(default)s，表示使用训练总步数）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每隔多少训练步保存一次中途检查点（默认：%(default)s）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="每个训练批次的序列数量（默认：%(default)s）",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=64,
        help="每条输入序列的 token 数量（默认：%(default)s）",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="模型隐藏向量的维度，必须能被注意力头数整除（默认：%(default)s）",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="堆叠的 Transformer Block 数量（默认：%(default)s）",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="每层的注意力头数，每个头的维度必须为偶数（默认：%(default)s）",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=192,
        help="SwiGLU 前馈网络的内部维度，通常约为 d_model 的 8/3 倍并取附近的 64 倍数（默认：%(default)s）",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE 旋转频率的基数参数，必须为正数（默认：%(default)s）",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-3,
        help="最大学习率，必须为正数（默认：%(default)s）",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-4,
        help="最小学习率，不能为负数或超过最大学习率（默认：%(default)s）",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="学习率预热步数，不能为负数（默认：%(default)s）",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="每隔多少训练步进行一次验证（默认：%(default)s）",
    )

    args = parser.parse_args()

    if args.warmup_steps < 0:
        parser.error("--warmup-steps 不能为负数")

    if args.total_steps <= args.warmup_steps:
        parser.error("--total-steps 必须大于预热步数")

    if args.lr_decay_steps is None:
        args.lr_decay_steps = args.total_steps

    if args.lr_decay_steps <= args.warmup_steps:
        parser.error("--lr-decay-steps 必须大于预热步数")

    if args.max_lr <= 0:
        parser.error("--max-lr 必须为正数")

    if not 0 <= args.min_lr <= args.max_lr:
        parser.error("--min-lr 必须在 0 和最大学习率之间")

    if args.eval_every <= 0:
        parser.error("--eval-every 必须为正数")

    if args.save_every <= 0:
        parser.error("--save-every 必须为正数")

    for name in (
        "batch_size", "context_length", "d_model",
        "num_layers", "num_heads", "d_ff",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} 必须为正数")

    if args.d_model % args.num_heads != 0:
        parser.error("d_model 必须能被 num_heads 整除")

    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE 要求每个注意力头的维度为偶数")

    if args.rope_theta <= 0:
        parser.error("rope_theta 必须为正数")

    return args


def save_config(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    """保存与检查点对应的命令行配置。

    :param checkpoint_path: 检查点路径。
    :param args: 已解析的训练参数。
    :return: 无返回值。
    """
    config_path = checkpoint_path.with_suffix(".config.json")

    config_path.write_text(
        json.dumps(
            vars(args),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def validate_resume_config(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    """核对模型配置，兼容没有配置文件的旧检查点。

    :param checkpoint_path: 要恢复的检查点路径。
    :param args: 当前训练参数。
    :return: 无返回值，配置不一致时抛出异常。
    """
    config_path = checkpoint_path.with_suffix(".config.json")

    if not config_path.exists():
        return

    saved = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    for name in (
        "context_length", "d_model", "num_layers",
        "num_heads", "d_ff", "rope_theta",
    ):
        if saved[name] != getattr(args, name):
            raise ValueError(
                f"模型配置不一致：{name}，"
                f"存档为 {saved[name]}，"
                f"当前为 {getattr(args, name)}"
            )


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int = 10,
) -> float:
    """在固定验证样本上计算平均损失，并恢复原来的训练模式。

    :param model: 要评估的语言模型。
    :param dataset: 验证集 token ID 数组。
    :param batch_size: 每个验证批次的序列数量。
    :param context_length: 每条序列的长度。
    :param device: 计算设备。
    :param eval_batches: 用于估计损失的批次数量，必须为正。
    :return: 验证批次的平均交叉熵损失。
    """
    if eval_batches <= 0:
        raise ValueError("eval_batches 必须为正")

    was_training = model.training
    random_state = np.random.get_state()

    model.eval()
    np.random.seed(1234)

    try:
        total_loss = 0.0

        for _ in range(eval_batches):
            input_ids, target_ids = get_batch(
                dataset,
                batch_size,
                context_length,
                device,
            )

            logits = model(input_ids)
            loss = cross_entropy(logits, target_ids)
            total_loss += loss.item()

        return total_loss / eval_batches
    finally:
        np.random.set_state(random_state)
        model.train(was_training)
