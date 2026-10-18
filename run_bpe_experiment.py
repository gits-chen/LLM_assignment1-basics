import json
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe, save_bpe


def main() -> None:
    """训练样本分词器，保存训练结果和实验指标。

    :return: 无返回值，结果写入配置的输出目录。
    """
    input_path = Path("data/owt_train.txt")
    output_dir = Path("artifacts/tokenizers/owt_32000")
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    elapsed = time.perf_counter() - start
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    longest_token = max(vocab.values(), key=len)

    report = {
        "input_path": str(input_path),
        "target_vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "training_seconds": elapsed,
        "peak_rss_mib": peak_rss_mib,
        "longest_token_bytes": len(longest_token),
        "longest_token_text": longest_token.decode(
            "utf-8", errors="replace"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    save_bpe(
        vocab,
        merges,
        output_dir / "vocab.json",
        output_dir / "merges.json",
    )

    with open(output_dir / "report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
