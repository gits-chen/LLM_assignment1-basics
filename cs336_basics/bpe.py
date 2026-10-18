import os
import regex
import json

def read_text(input_path: str | os.PathLike) -> str:
    """读取 UTF-8 文本文件，保留原有空格和换行。

    :param input_path: 训练语料的文件路径。
    :return: 文件的全部文本内容。
    """
    with open(input_path, "r", encoding="utf-8", newline="") as file:
        text = file.read()
    return text

def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    chunk_size: int = 1024 * 1024,
) -> dict[tuple[bytes, ...], int]:
    """按字符块读取文件，在确认完整的特殊 token 边界处分批预分词。

    :param input_path: UTF-8 训练语料的文件路径。
    :param special_tokens: 用作预分词边界的特殊 token，元素不能为空字符串。
    :param chunk_size: 每次读取的字符数量，必须为正整数。
    :return: 初始字节序列到出现次数的累计字典。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if not special_tokens:
        return pretokenize(read_text(input_path), special_tokens)

    if any(token == "" for token in special_tokens):
        raise ValueError("特殊 token 不能为空字符串")

    pattern = regex.compile(
        "|".join(
            regex.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
    )
    longest_special = max(len(token) for token in special_tokens)
    counts: dict[tuple[bytes, ...], int] = {}
    buffer = ""

    with open(input_path, "r", encoding="utf-8", newline="") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            buffer += chunk
            safe_end = len(buffer) - longest_special
            cut = 0

            for match in pattern.finditer(buffer):
                if match.end() > safe_end:
                    break
                cut = match.end()

            if cut == 0:
                continue

            part_counts = pretokenize(buffer[:cut], special_tokens)

            for sequence, frequency in part_counts.items():
                counts[sequence] = counts.get(sequence, 0) + frequency

            buffer = buffer[cut:]

    part_counts = pretokenize(buffer, special_tokens)

    for sequence, frequency in part_counts.items():
        counts[sequence] = counts.get(sequence, 0) + frequency

    return counts

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

def build_pair_index(
    token_sequences: list[tuple[bytes, ...]],
) -> dict[tuple[bytes, bytes], set[int]]:
    """建立相邻 token 对到片段编号集合的索引。

    :param token_sequences: 不同预分词片段组成的列表，列表下标作为固定编号。
    :return: 每个相邻 token 对对应的片段编号集合。
    """
    pair_index: dict[tuple[bytes, bytes], set[int]] = {}

    for sequence_id, token_sequence in enumerate(token_sequences):
        for i in range(len(token_sequence) - 1):
            pair = (token_sequence[i], token_sequence[i + 1])

            if pair not in pair_index:
                pair_index[pair] = set()

            pair_index[pair].add(sequence_id)

    return pair_index

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

def merge_sequence(
    token_sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """在单个片段内从左到右合并指定相邻对，匹配位置不重叠。

    :param token_sequence: 合并前的 token 序列。
    :param pair: 要合并的两个相邻 token。
    :return: 合并后的 token 序列。
    """
    new_tokens: list[bytes] = []
    merged_token = pair[0] + pair[1]
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

    return tuple(new_tokens)

def merge_pair(
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """在所有片段内应用指定合并规则，并累计相同结果的频率。

    :param pretoken_counts: 合并前的 token 序列及其出现次数。
    :param pair: 本轮需要合并的两个相邻 token。
    :return: 合并后的 token 序列到出现次数的字典。
    """
    new_counts: dict[tuple[bytes, ...], int] = {}

    for token_sequence, frequency in pretoken_counts.items():
        new_sequence = merge_sequence(token_sequence, pair)
        new_counts[new_sequence] = new_counts.get(new_sequence, 0) + frequency

    return new_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """利用片段索引和增量频率统计训练 BPE。

    :param input_path: UTF-8 训练语料的文件路径。
    :param vocab_size: 包含基础字节和特殊 token 的目标词表大小。
    :param special_tokens: 加入词表并作为训练分段边界的特殊 token 列表。
    :param kwargs: 为兼容测试入口保留的额外关键字参数。
    :return: 词表和按学习顺序排列的合并规则。
    """
    vocab = initialize_vocab(special_tokens)

    if vocab_size < len(vocab):
        raise ValueError("目标词表大小不能小于初始词表大小")

    pretoken_counts = pretokenize_file(input_path, special_tokens)

    token_sequences = list(pretoken_counts.keys())
    frequencies = list(pretoken_counts.values())
    pair_counts = count_pairs(pretoken_counts)

    del pretoken_counts

    pair_index = build_pair_index(token_sequences)

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair = select_pair(pair_counts)

        if pair is None:
            break

        affected_ids = list(pair_index[pair])

        for sequence_id in affected_ids:
            old_sequence = token_sequences[sequence_id]
            frequency = frequencies[sequence_id]
            old_counts = count_pairs({old_sequence: frequency})

            for old_pair, count in old_counts.items():
                pair_counts[old_pair] -= count

                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]

                pair_index[old_pair].remove(sequence_id)

                if not pair_index[old_pair]:
                    del pair_index[old_pair]

            new_sequence = merge_sequence(old_sequence, pair)
            token_sequences[sequence_id] = new_sequence
            new_counts = count_pairs({new_sequence: frequency})

            for new_pair, count in new_counts.items():
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count

                if new_pair not in pair_index:
                    pair_index[new_pair] = set()

                pair_index[new_pair].add(sequence_id)

        vocab[len(vocab)] = pair[0] + pair[1]
        merges.append(pair)

    return vocab, merges

def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> None:
    """将词表和有序合并规则保存为使用字节值列表的 JSON 文件。

    :param vocab: 整数 token ID 到字节串的词表。
    :param merges: 按学习顺序排列的合并规则。
    :param vocab_filepath: 词表输出路径。
    :param merges_filepath: 合并规则输出路径。
    :return: 无返回值，结果写入指定文件。
    """
    vocab_data: dict[str, list[int]] = {}

    for token_id, token_bytes in vocab.items():
        vocab_data[str(token_id)] = list(token_bytes)

    merges_data: list[list[list[int]]] = []

    for left, right in merges:
        merges_data.append([list(left), list(right)])

    with open(vocab_filepath, "w", encoding="utf-8") as file:
        json.dump(vocab_data, file, indent=2)

    with open(merges_filepath, "w", encoding="utf-8") as file:
        json.dump(merges_data, file, indent=2)