from __future__ import annotations

import os
import regex
import json

from collections.abc import Iterable, Iterator


class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]
    bytes_to_id: dict[bytes, int]
    merge_ranks: dict[tuple[bytes, bytes], int]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """复制词表和合并规则，补充特殊 token，并建立编码查询表。

        :param vocab: 整数 token ID 到完整字节串的词表。
        :param merges: 按学习顺序排列的相邻 token 合并规则。
        :param special_tokens: 编码时完整保留的特殊 token 列表，None 表示不额外指定。
        :return: 无返回值，仅初始化当前对象。
        """
        self.vocab = vocab.copy()
        self.merges = merges.copy()

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens.copy()

        self.bytes_to_id = {}

        for token_id, token_byte in self.vocab.items():
            self.bytes_to_id[token_byte] = token_id

        next_id = max(self.vocab, default=-1) + 1

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id += 1

        self.merge_ranks = {}

        for rank, pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank


    def decode(self, ids: list[int]) -> str:
        """拼接各 ID 对应的字节并解码为文字，无效 UTF-8 使用 U+FFFD 替换。

        :param ids: 按文本顺序排列的 token ID 列表。
        :return: 解码后的 Unicode 字符串；空列表得到空字符串。
        """
        byte_parts = []

        for token_id in ids:
            token_bytes = self.vocab[token_id]
            byte_parts.append(token_bytes)

        data = b"".join(byte_parts)

        return data.decode("utf-8", errors="replace")

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """按已有合并规则的优先级编码一个普通预分词片段。

        :param pretoken: 保留原有空格和标点的普通文本片段。
        :return: 该片段对应的 token ID 列表。
        """
        tokens: list[bytes] = []

        for byte_value in pretoken.encode("utf-8"):
            tokens.append(bytes([byte_value]))

        while len(tokens) > 1:
            best_position = None
            best_rank = len(self.merges)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_position = i
                    best_rank = rank

            if best_position is None:
                break

            i = best_position
            merged_token = tokens[i] + tokens[i + 1]
            tokens[i:i + 2] = [merged_token]

        ids: list[int] = []

        for token in tokens:
            ids.append(self.bytes_to_id[token])

        return ids

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """对普通文本预分词，逐段编码并按原顺序拼接结果。

        :param text: 已排除指定特殊 token 的普通文本。
        :return: 按输入顺序排列的 token ID 列表。
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        ids: list[int] = []

        for match in regex.finditer(PAT, text):
            pretoken = match.group()

            pretoken_ids = self._encode_pretoken(pretoken)

            ids.extend(pretoken_ids)

        return ids

    def encode(self, text: str) -> list[int]:
        """编码完整文本，并将指定的特殊 token 保留为独立 ID。

        :param text: 待编码的 Unicode 字符串。
        :return: 按文本顺序排列的 token ID 列表；空文本得到空列表。
        """
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        escaped_tokens = []

        for token in sorted(self.special_tokens, key=len, reverse=True):
            escaped_tokens.append(regex.escape(token))

        pattern = "(" + "|".join(escaped_tokens) + ")"
        parts = regex.split(pattern, text)

        ids: list[int] = []

        for part in parts:
            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                token_id = self.bytes_to_id[token_bytes]
                ids.append(token_id)
            else:
                ids.extend(self._encode_ordinary_text(part))

        return ids

    def _iter_text_segments(
        self,
        text: str,
    ) -> Iterator[tuple[str, bool, int]]:
        """按原文顺序逐个产生普通片段和完整特殊 token 的描述。

        :param text: 本次待分析的文本，例如流式编码的当前缓冲区。
        :return: 生成器，每项为 (片段文字, 是否特殊 token, 不包含该位置的结束下标)；
            下标按输入 text 的字符串位置计算，不是 UTF-8 字节偏移。
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        position = 0

        if self.special_tokens:
            escaped_tokens = []

            for token in sorted(
                self.special_tokens, key=len, reverse=True
            ):
                escaped_tokens.append(regex.escape(token))

            pattern = "|".join(escaped_tokens)

            for special in regex.finditer(pattern, text):
                ordinary = text[position:special.start()]

                for match in regex.finditer(PAT, ordinary):
                    yield (
                        match.group(),
                        False,
                        position + match.end(),
                    )

                yield special.group(), True, special.end()
                position = special.end()

        for match in regex.finditer(PAT, text[position:]):
            yield match.group(), False, position + match.end()

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """缓冲跨块的未确定文本，逐步编码并输出已确定的 token ID。

        :param iterable: 逐个提供字符串的可迭代对象，例如打开的文本文件。
        :return: 按文本顺序逐个产生整数 token ID 的生成器。
        """
        buffer = ""
        longest_special = 0

        for token in self.special_tokens:
            longest_special = max(longest_special, len(token))

        for chunk in iterable:
            buffer += chunk
            safe_end = len(buffer) - longest_special
            recent = []
            consumed = 0

            for segment in self._iter_text_segments(buffer):
                recent.append(segment)

                if len(recent) <= 2:
                    continue

                piece, is_special, end = recent.pop(0)

                if end > safe_end:
                    break

                if is_special:
                    yield self.bytes_to_id[piece.encode("utf-8")]
                else:
                    yield from self._encode_pretoken(piece)

                consumed = end
            buffer = buffer[consumed:]

        yield from self.encode(buffer)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """从使用字节值列表保存的 JSON 文件加载分词器。

        :param vocab_filepath: 词表文件路径，内容为字符串 ID 到字节值列表的映射。
        :param merges_filepath: 合并规则文件路径，每项包含两个字节值列表。
        :param special_tokens: 额外指定的特殊 token 列表。
        :return: 使用加载结果创建的分词器对象。
        """
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab_data = json.load(file)

        vocab: dict[int, bytes] = {}

        for token_id, byte_values in vocab_data.items():
            vocab[int(token_id)] = bytes(byte_values)

        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges_data = json.load(file)

        merges: list[tuple[bytes, bytes]] = []

        for left, right in merges_data:
            merges.append((bytes(left), bytes(right)))

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )
