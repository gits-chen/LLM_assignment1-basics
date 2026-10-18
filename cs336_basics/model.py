import torch
from torch import nn

from cs336_basics.layers import (
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    Embedding,
    Linear,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建使用 RoPE 的 Pre-Norm Transformer Block。

        :param d_model: 输入和输出的特征维度。
        :param num_heads: 注意力头数量。
        :param d_ff: 前馈网络内部的特征维度。
        :param max_seq_len: RoPE 缓存支持的最大序列长度。
        :param theta: RoPE 的频率参数。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须为正数且能整除 d_model")

        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.ln1 = RMSNorm(
            d_model, device=device, dtype=dtype
        )
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            device=device,
            dtype=dtype,
            rope=rope,
        )

        self.ln2 = RMSNorm(
            d_model, device=device, dtype=dtype
        )
        self.ffn = SwiGLU(
            d_model, d_ff, device=device, dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """依次执行带残差连接的注意力和前馈计算。

        :param x: 形状为 (..., seq_len, d_model) 的输入。
        :param token_positions: token 的位置编号，默认连续编号。
        :return: 形状与输入相同的输出张量。
        """
        attention_output = self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )
        y = x + attention_output

        ffn_output = self.ffn(self.ln2(y))
        return y + ffn_output

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建基于 Transformer 的因果语言模型。

        :param vocab_size: 词表大小。
        :param context_length: 模型支持的最大输入序列长度。
        :param d_model: token 表示的特征维度。
        :param num_layers: Transformer Block 的数量。
        :param num_heads: 每个 Block 的注意力头数量。
        :param d_ff: 前馈网络内部的特征维度。
        :param rope_theta: RoPE 的频率参数。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """将 token ID 序列转换为下一 token 的预测分数。

        :param token_ids: 形状为 (batch_size, seq_len) 的整数张量。
        :return: 形状为 (batch_size, seq_len, vocab_size) 的 logits。
        """
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("输入序列长度超过 context_length")

        x = self.token_embeddings(token_ids)

        for block in self.layers:
            x = block(x)

        x = self.ln_final(x)

        return self.lm_head(x)