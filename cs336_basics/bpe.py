import os
import regex


def read_text(input_path: str | os.PathLike) -> str:
    """读取 UTF-8 文本文件，保留原有空格和换行。

    :param input_path: 训练语料的文件路径。
    :return: 文件的全部文本内容。
    """
    with open(input_path, "r", encoding="utf-8", newline="") as file:
        text = file.read()
    return text

def pretokenize(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """隔离特殊 token，将普通文本预分词并统计初始字节序列的频率。

    :param text: 待处理的 Unicode 文本。
    :param special_tokens: 作为分段边界、不参与频率统计的特殊 token 列表。
    :return: 单字节 token 元组到片段出现次数的字典。
    """
    if special_tokens:
        escaped_tokens = []

        for token in sorted(special_tokens, key=len, reverse=True):
            escaped_tokens.append(regex.escape(token))

        special_pattern = "|".join(escaped_tokens)
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    counts = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for part in parts:
        for match in regex.finditer(PAT, part):
                word = match.group()

                encoded_word = word.encode("utf-8")
                byte_tokens = []

                for byte_value in encoded_word:
                    byte_tokens.append(bytes([byte_value]))

                token_sequence = tuple(byte_tokens)

                counts[token_sequence] = counts.get(token_sequence, 0) + 1

    return counts


def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """建立包含 256 个基础字节和特殊 token 的初始词表。

    :param special_tokens: 按顺序追加的特殊 token 列表，预期元素不重复。
    :return: 整数 token ID 到完整字节串的字典。
    """
    vocab = {}

    for byte_value in range(256):
        vocab[byte_value] = bytes([byte_value])

    for special_token in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = special_token.encode("utf-8")

    return vocab


def count_pairs(
    pretoken_counts: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """按片段频率加权统计各片段内部的相邻 token 对。

    :param pretoken_counts: 当前 token 序列到片段出现次数的映射。
    :return: 相邻 token 对到总出现次数的字典。
    """
    pair_counts = {}

    for token_sequence, frequency in pretoken_counts.items():
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i], token_sequence[i + 1])

            pair_counts[pair] = pair_counts.get(pair, 0) + frequency

    return pair_counts

def select_pair(
    pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    """选择最高频的相邻对，频率相同时选择字典序更大的对。

    :param pair_counts: 相邻 token 对到正整数出现次数的映射。
    :return: 选中的 token 对；输入为空时返回 None。
    """
    if not pair_counts:
        return None

    best_pair = None
    best_frequency = -1

    for pair, frequency in pair_counts.items():
        if frequency > best_frequency:
            best_pair = pair
            best_frequency = frequency

        elif frequency == best_frequency:
            if best_pair is not None and pair > best_pair:
                best_pair = pair

    return best_pair


def merge_pair(
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """在各片段内从左到右应用指定合并规则，匹配位置不重叠。

    :param pretoken_counts: 合并前的 token 序列及其片段出现次数。
    :param pair: 本轮需要合并的两个相邻 bytes token。
    :return: 合并后序列到出现次数的新字典，相同结果序列的频率累计。
    """
    new_counts = {}
    merged_token = pair[0] + pair[1]

    for token_sequence, frequency in pretoken_counts.items():
        new_tokens = []
        i = 0

        while i < len(token_sequence):
            if (
                i + 1 < len(token_sequence)
                and token_sequence[i] == pair[0]
                and token_sequence[i + 1] == pair[1]
            ):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(token_sequence[i])
                i += 1

        new_sequence = tuple(new_tokens)
        new_counts[new_sequence] = new_counts.get(new_sequence, 0) + frequency

    return new_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从文本语料训练 BPE，达到目标容量或没有可合并对时停止。

    :param input_path: UTF-8 训练语料的文件路径。
    :param vocab_size: 词表大小上限，包含基础字节、特殊 token 和合并得到的 token。
    :param special_tokens: 加入词表并作为训练分段边界的特殊 token 列表。
    :param kwargs: 为兼容测试入口保留的额外关键字参数，当前未使用。
    :return: (vocab, merges)，分别为 ID 到字节串的词表和按学习顺序排列的合并规则。
    """
    vocab = initialize_vocab(special_tokens)

    if vocab_size < len(vocab):
        raise ValueError("目标词表大小不能小于初始词表大小")

    text = read_text(input_path)
    pretoken_counts = pretokenize(text, special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        pair = select_pair(pair_counts)

        if pair is None:
            break

        pretoken_counts = merge_pair(pretoken_counts, pair)

        new_token = pair[0] + pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token

        merges.append(pair)

    return vocab, merges
