import math

import torch
from torch import nn
from einops import einsum, rearrange


def silu(x: torch.Tensor) -> torch.Tensor:
    """对输入逐元素应用 SiLU 激活函数。

    :param x: 输入浮点张量。
    :return: 形状与输入相同的张量。
    """
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿指定维度计算数值稳定的 Softmax。

    :param x: 输入浮点张量。
    :param dim: 进行归一化的维度。
    :return: 形状与输入相同，沿指定维度求和为 1 的张量。
    """
    max_values = x.max(dim=dim, keepdim=True).values
    exp_values = torch.exp(x - max_values)
    return exp_values / exp_values.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """使用 einops.einsum 计算缩放点积注意力。

    :param Q: 形状为 (..., queries, d_k) 的查询张量。
    :param K: 形状为 (..., keys, d_k) 的键张量。
    :param V: 形状为 (..., keys, d_v) 的值张量。
    :param mask: 布尔掩码，True 表示允许关注。
    :return: 形状为 (..., queries, d_v) 的注意力输出。
    """
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")

    scores = scores / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)

    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建不带偏置的线性层，并按作业要求初始化权重。

        :param in_features: 输入张量最后一个维度的大小。
        :param out_features: 输出张量最后一个维度的大小。
        :param device: 存放权重的设备。
        :param dtype: 权重的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入的最后一个维度应用线性变换。

        :param x: 形状为 (..., in_features) 的输入张量。
        :return: 形状为 (..., out_features) 的输出张量。
        """
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建 token ID 到向量的可训练查找表。

        :param num_embeddings: 词表大小，即向量表的行数。
        :param embedding_dim: 每个 token 对应向量的维度。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """按 token ID 查找对应的向量。

        :param token_ids: 取值在词表范围内的整数 ID 张量。
        :return: 在输入形状末尾增加 embedding_dim 的浮点张量。
        """
        return self.weight[token_ids]


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建使用 SwiGLU 的前馈网络。

        :param d_model: 输入和输出的特征维度。
        :param d_ff: 内部隐藏特征的维度。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        self.w1 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )
        self.w2 = Linear(
            d_ff, d_model, device=device, dtype=dtype
        )
        self.w3 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对每个位置的输入特征应用 SwiGLU 变换。

        :param x: 形状为 (..., d_model) 的浮点张量。
        :return: 形状与输入相同的浮点张量。
        """
        gate = silu(self.w1(x))
        features = self.w3(x)
        hidden = gate * features
        return self.w2(hidden)

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """创建 RMS 归一化层。

        :param d_model: 输入张量最后一个维度的大小。
        :param eps: 加入均方值的小常数，用于数值稳定。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :return: 无返回值。
        """
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """沿最后一个维度归一化，再应用逐维缩放。

        :param x: 形状为 (..., d_model) 的浮点张量。
        :return: 形状和数据类型与输入相同的张量。
        """
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        mean_square = x_float.square().mean(
            dim=-1,
            keepdim=True,
        )
        rms = torch.sqrt(mean_square + self.eps)
        output = x_float / rms * self.weight

        return output.to(input_dtype)

class  RotaryPositionalEmbedding(nn.Module):
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k 必须是偶数")

        pair_indices = torch.arange(
            d_k // 2,
            dtype=torch.float32,
            device=device,
        )

        exponents = 2 * pair_indices / d_k
        inv_freq = theta ** (-exponents)

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        angles = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """根据 token 位置旋转输入向量中的相邻元素对。

        :param x: 形状为 (..., seq_len, d_k) 的输入张量。
        :param token_positions: 位置编号，取值在缓存范围内。
        :return: 形状和数据类型与输入相同的旋转结果。
        """
        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        paired = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        return paired.flatten(start_dim=-2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        """创建带因果掩码的多头自注意力模块。

        :param d_model: 输入和输出的特征维度。
        :param num_heads: 注意力头数量，必须整除 d_model。
        :param device: 存放参数的设备。
        :param dtype: 参数的数据类型。
        :param rope: 可选的旋转位置编码模块。
        :return: 无返回值。
        """
        super().__init__()

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("num_heads 必须为正数且能整除 d_model")

        self.num_heads = num_heads
        self.rope = rope

        self.q_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None
        ) -> torch.Tensor:
        """生成多头 Q、K、V，并计算因果自注意力。

        :param x: 形状为 (..., seq_len, d_model) 的输入。
        :param token_positions: token 的位置编号，默认从 0 开始连续编号。
        :return: 形状与输入相同的注意力输出。
        """
        q = rearrange(
            self.q_proj(x),
            "... seq (heads d) -> ... heads seq d",
            heads=self.num_heads,
        )
        k = rearrange(
            self.k_proj(x),
            "... seq (heads d) -> ... heads seq d",
            heads=self.num_heads,
        )
        v = rearrange(
            self.v_proj(x),
            "... seq (heads d) -> ... heads seq d",
            heads=self.num_heads,
        )

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2],
                    device=x.device,
                )

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
            
        seq_len = x.shape[-2]
        mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device,
        ).tril()

        output = scaled_dot_product_attention(
            q, k, v, mask=mask
        )

        output = rearrange(
            output,
            "... heads seq d -> ... seq (heads d)",
        )

        return self.output_proj(output)