import json
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_file(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 65536,
) -> dict[str, str | int | float]:
    """流式编码文本，将 token ID 分批保存为小端 uint16 原始二进制文件。

    :param tokenizer: 用于编码的分词器。
    :param input_path: UTF-8 文本文件路径。
    :param output_path: 输出的二进制文件路径。
    :param batch_size: 每批写入的 token 数量，必须大于零。
    :return: 输入输出路径、数据类型、token 数量、文件大小和处理耗时。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    if any(token_id < 0 or token_id > 65535 for token_id in tokenizer.vocab):
        raise ValueError("词表 ID 超出了 uint16 的范围")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入文件和输出文件不能相同")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    token_count = 0
    start = time.perf_counter()

    with open(input_path, "r", encoding="utf-8", newline="") as source:
        with open(output_path, "wb") as target:
            for token_id in tokenizer.encode_iterable(source):
                buffer.append(token_id)
                token_count += 1

                if len(buffer) >= batch_size:
                    np.asarray(buffer, dtype="<u2").tofile(target)
                    buffer.clear()

            if buffer:
                np.asarray(buffer, dtype="<u2").tofile(target)

    elapsed = time.perf_counter() - start

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "<u2",
        "token_count": token_count,
        "output_bytes": output_path.stat().st_size,
        "processing_seconds": elapsed,
    }

    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return report


def main() -> None:
    """使用 OpenWebText 分词器编码正式验证集和训练集。

    :return: 无返回值，保存二进制数据、元信息并打印处理报告。
    """
    tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/tokenizers/owt_32000/vocab.json",
        merges_filepath="artifacts/tokenizers/owt_32000/merges.json",
        special_tokens=["<|endoftext|>"],
    )

    jobs = [
        (
            Path("data/owt_valid.txt"),
            Path("artifacts/encoded/owt_valid.bin"),
        ),
        (
            Path("data/owt_train.txt"),
            Path("artifacts/encoded/owt_train.bin"),
        ),
    ]

    for input_path, output_path in jobs:
        print(f"开始编码：{input_path}", flush=True)

        report = encode_file(
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
        )

        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
