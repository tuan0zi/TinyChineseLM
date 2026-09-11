import json
from pathlib import Path

from datasets import load_dataset


# =========================
# 配置
# =========================

OUTPUT_PATH = Path(
    "data/tokenizer/tokenizer_train.jsonl"
)

FINEWEB_TARGET_MB = 160
WIKI_TARGET_MB = 40

MIN_TEXT_LENGTH = 200

MB = 1024 * 1024


def clean_text(text: str):

    if not isinstance(text, str):
        return None

    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH:
        return None

    return text


def write_source(
    dataset,
    source_name,
    target_bytes,
    fout
):

    written_bytes = 0
    document_count = 0

    for item in dataset:

        text = clean_text(
            item["text"]
        )

        if text is None:
            continue

        record = {
            "text": text,
            "source": source_name
        }

        line = (
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

        line_bytes = len(
            line.encode("utf-8")
        )

        if (
            written_bytes
            + line_bytes
            > target_bytes
        ):
            break

        fout.write(line)

        written_bytes += line_bytes
        document_count += 1

        if document_count % 1000 == 0:

            print(
                f"{source_name} | "
                f"documents={document_count} | "
                f"size={written_bytes / MB:.2f} MB"
            )

    print(
        f"{source_name} finished | "
        f"documents={document_count} | "
        f"size={written_bytes / MB:.2f} MB"
    )


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================
    # FineWeb2 中文大陆简体
    # =========================

    fineweb = load_dataset(
        "agentlans/fineweb2-chinese",
        "MAINLAND_CHINA",
        split="train",
        streaming=True
    )

    fineweb = fineweb.shuffle(
        seed=42,
        buffer_size=10000
    )


    # =========================
    # 中文 Wikipedia
    # =========================

    wiki = load_dataset(
        "yuhuanstudio/wikipedia-zh",
        split="train",
        streaming=True
    )

    wiki = wiki.shuffle(
        seed=42,
        buffer_size=10000
    )


    # =========================
    # 写入统一 JSONL
    # =========================

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as fout:

        write_source(
            dataset=fineweb,
            source_name="fineweb2",
            target_bytes=(
                FINEWEB_TARGET_MB * MB
            ),
            fout=fout
        )

        write_source(
            dataset=wiki,
            source_name="wikipedia",
            target_bytes=(
                WIKI_TARGET_MB * MB
            ),
            fout=fout
        )

    print(
        "Tokenizer corpus saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()