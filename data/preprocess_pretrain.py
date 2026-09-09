import json
from pathlib import Path

from tqdm import tqdm


def split_text(
    text: str,
    chunk_size: int = 512
) -> list[str]:
    """
    暂时按字符数切分文本。

    后续会升级为基于 token 的 packing。
    """
    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks   #chunk 是字符 这里是 512个字符 和 512个token的概念不太一样


def preprocess_pretrain_data(
    input_path: str,
    output_path: str,
    chunk_size: int = 512
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as src, open(
        output_path,
        "w",
        encoding="utf-8"
    ) as dst:

        for line_num, line in enumerate(
            tqdm(src, desc="Processing"),
            start=1
        ):
            try:
                data = json.loads(line)

                text = data["text"].strip()

                if not text:
                    continue

                chunks = split_text(
                    text,
                    chunk_size
                )

                for chunk in chunks:
                    record = {
                        "text": chunk
                    }

                    dst.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

            except json.JSONDecodeError:
                print(
                    f"Invalid JSON at line "
                    f"{line_num}"
                )

            except KeyError:
                print(
                    f"Missing 'text' at line "
                    f"{line_num}"
                )


if __name__ == "__main__":
    preprocess_pretrain_data(
        input_path=(
            "data/raw/"
            "test_pretrain.jsonl"
        ),
        output_path=(
            "data/processed/"
            "pretrain.jsonl"
        ),
        chunk_size=512
    )