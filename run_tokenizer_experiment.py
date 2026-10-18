from pathlib import Path
import json
import time

from cs336_basics.tokenizer import Tokenizer

def measure_throughput(
    tokenizer: Tokenizer,
    documents: list[str],
) -> dict[str, int | float]:
    """测量内存中文档的编码吞吐量，并估算编码 825 GB 文本的耗时。

    :param tokenizer: 用于编码的分词器。
    :param documents: 已读取的文档正文列表。
    :return: 样本规模、编码耗时、吞吐量和预计处理时间。
    """
    total_bytes = sum(
        len(document.encode("utf-8"))
        for document in documents
    )

    if total_bytes == 0:
        raise ValueError("样本文本不能为空")

    tokenizer.encode(documents[0])

    total_tokens = 0
    start = time.perf_counter()

    for document in documents:
        ids = tokenizer.encode(document)
        total_tokens += len(ids)

    elapsed = time.perf_counter() - start
    bytes_per_second = total_bytes / elapsed
    estimated_pile_seconds = 825 * 10**9 / bytes_per_second

    return {
        "document_count": len(documents),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "encoding_seconds": elapsed,
        "bytes_per_second": bytes_per_second,
        "mib_per_second": bytes_per_second / (1024 * 1024),
        "estimated_pile_hours": estimated_pile_seconds / 3600,
    }

def read_documents(
    input_path: str | Path,
    count: int = 10,
) -> list[str]:
    """读取文件开头指定数量的非空文档，保留正文中的空白字符。

    :param input_path: UTF-8 文本文件路径。
    :param count: 需要读取的文档数量，必须大于零。
    :return: 文档正文列表，不包含文档分隔符。
    """
    if count <= 0:
        raise ValueError("count 必须大于 0")

    documents: list[str] = []
    separator = "<|endoftext|>"
    buffer = ""

    with open(input_path, "r", encoding="utf-8", newline="") as file:
        while len(documents) < count:
            chunk = file.read(1024 * 1024)

            if not chunk:
                if buffer.strip():
                    documents.append(buffer)
                break

            buffer += chunk
            parts = buffer.split(separator)
            buffer = parts.pop()

            for document in parts:
                if document.strip():
                    documents.append(document)

                if len(documents) == count:
                    return documents

    return documents

def measure_compression(
    tokenizer: Tokenizer,
    documents: list[str],
) -> dict[str, int | float]:
    """编码文档，检查解码结果，并统计平均每个 token 表示的字节数。

    :param tokenizer: 用于编码和解码的分词器。
    :param documents: 不包含文档分隔符的正文列表。
    :return: 文档数量、总字节数、总 token 数和 bytes/token。
    """
    total_bytes = 0
    total_tokens = 0

    for document in documents:
        ids = tokenizer.encode(document)

        if tokenizer.decode(ids) != document:
            raise ValueError("编码后解码的结果与原文不一致")

        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(ids)

    if total_tokens == 0:
        raise ValueError("样本没有产生 token，无法计算压缩率")

    return {
        "document_count": len(documents),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }

def main() -> None:
    """加载两套分词器，对固定文档样本进行压缩率比较。

    :return: 无返回值，实验结果以 JSON 打印。
    """
    tinystories_tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/tokenizers/tinystories_10000/vocab.json",
        merges_filepath="artifacts/tokenizers/tinystories_10000/merges.json",
        special_tokens=["<|endoftext|>"],
    )

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="artifacts/tokenizers/owt_32000/vocab.json",
        merges_filepath="artifacts/tokenizers/owt_32000/merges.json",
        special_tokens=["<|endoftext|>"],
    )

    tinystories_documents = read_documents(
        "data/TinyStoriesV2-GPT4-train.txt",
        count=10,
    )

    owt_documents = read_documents(
        "data/owt_train.txt",
        count=10,
    )

    results = {
        "tinystories_on_tinystories": measure_compression(
            tinystories_tokenizer,
            tinystories_documents,
        ),
        "owt_on_owt": measure_compression(
            owt_tokenizer,
            owt_documents,
        ),
        "tinystories_on_owt": measure_compression(
            tinystories_tokenizer,
            owt_documents,
        ),
    }

    tinystories_speed_documents = read_documents(
        "data/TinyStoriesV2-GPT4-train.txt",
        count=1000,
    )

    owt_speed_documents = read_documents(
        "data/owt_train.txt",
        count=1000,
    )

    results["tinystories_throughput"] = measure_throughput(
        tinystories_tokenizer,
        tinystories_speed_documents,
    )

    results["owt_throughput"] = measure_throughput(
        owt_tokenizer,
        owt_speed_documents,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
