import json
import hashlib
from pathlib import Path

import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# 1. 基本配置
# ============================================================

TOKENIZER_PATH = "tokenizer/tokenizer_8k"

OUTPUT_DIR = Path(
    "data/packed"
)

SEQ_LEN = 512


# ============================================================
# 2. Token 数目标
# ============================================================

# 正式训练集：1亿 token
TRAIN_TARGET_TOKENS = 100_000_000

# 验证集：200万 token
VAL_TARGET_TOKENS = 2_000_000


# FineWeb : Wikipedia = 80 : 20

FINEWEB_TRAIN_TARGET = 80_000_000
WIKI_TRAIN_TARGET = 20_000_000

FINEWEB_VAL_TARGET = 1_600_000
WIKI_VAL_TARGET = 400_000


MIN_TEXT_LENGTH = 200   #文本清洗 小于该长度的 认为 没有有用信息，直接删掉

VAL_PERCENT = 2


# ============================================================
# 3. 文本清洗
# ============================================================

def clean_text(text: str):

    if not isinstance(text, str):
        return None

    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH:
        return None

    return text


# ============================================================
# 4. 判断一个 document 属于 train 还是 val
# ============================================================

def is_validation_document(text: str):   #确定性划分训练集和验证集

    digest = hashlib.md5(  #该函数的输入是字节 不能是字符串 ，每一个文档输出对应的一个数字指纹
        text.encode("utf-8")
    ).digest()# 得到原始的二进制哈希值

    value = int.from_bytes(
        digest[:4],
        byteorder="big"
    )

    return (
        value % 100
        <
        VAL_PERCENT
    )


# ============================================================
# 5. 写 token 到二进制文件
# ============================================================

def write_tokens(
    fout,
    token_ids,
    max_tokens
):

    if max_tokens <= 0:
        return 0

    token_ids = token_ids[:max_tokens]

    array = np.asarray(
        token_ids,
        dtype=np.uint16
    )

    fout.write(
        array.tobytes()
    )

    return len(token_ids)


# ============================================================
# 6. 处理一个数据源
# ============================================================

def process_source(
    dataset,
    source_name,
    tokenizer,
    train_target,
    val_target,
    train_file,
    val_file
):

    train_tokens = 0
    val_tokens = 0

    train_documents = 0
    val_documents = 0

    scanned_documents = 0


    for item in dataset:

        if (
            train_tokens >= train_target
            and
            val_tokens >= val_target
        ):
            break


        text = clean_text(
            item["text"]
        )

        if text is None:
            continue


        scanned_documents += 1


        # --------------------------------
        # 文档级 Train / Validation 划分
        # --------------------------------

        is_val = is_validation_document(
            text
        )


        # 如果这一边已经够了，就跳过
        if is_val:

            if val_tokens >= val_target:
                continue

        else:

            if train_tokens >= train_target:
                continue


        # --------------------------------
        # 文本 → Token IDs
        # --------------------------------

        ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )


        # 每篇文档后面加入 EOS
        ids.append(
            tokenizer.eos_token_id
        )


        # --------------------------------
        # 写 Validation
        # --------------------------------

        if is_val:

            remaining = (
                val_target
                -
                val_tokens
            )

            written = write_tokens(
                fout=val_file,
                token_ids=ids,
                max_tokens=remaining
            )

            val_tokens += written

            if written > 0:
                val_documents += 1


        # --------------------------------
        # 写 Train
        # --------------------------------

        else:

            remaining = (
                train_target
                -
                train_tokens
            )

            written = write_tokens(
                fout=train_file,
                token_ids=ids,
                max_tokens=remaining
            )

            train_tokens += written

            if written > 0:
                train_documents += 1


        # --------------------------------
        # 打印进度
        # --------------------------------

        if scanned_documents % 5000 == 0:

            print(
                f"{source_name} | "
                f"scanned={scanned_documents} | "
                f"train={train_tokens:,}/{train_target:,} | "
                f"val={val_tokens:,}/{val_target:,}"
            )


    print(
        f"\n{source_name} finished"
    )

    print(
        f"train tokens: {train_tokens:,}"
    )

    print(
        f"val tokens: {val_tokens:,}"
    )

    print(
        f"train documents: {train_documents:,}"
    )

    print(
        f"val documents: {val_documents:,}"
    )


    return {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_documents": train_documents,
        "val_documents": val_documents
    }


# ============================================================
# 7. 主函数
# ============================================================

def main():

    # --------------------------------
    # 创建输出目录
    # --------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------
    # 加载正式 Tokenizer
    # --------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True
    )


    print(
        "Tokenizer vocab size:",
        len(tokenizer)
    )


    assert len(tokenizer) <= 65535

    assert tokenizer.eos_token_id is not None


    # --------------------------------
    # 加载 FineWeb2
    # --------------------------------

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


    # --------------------------------
    # 加载 Wikipedia
    # --------------------------------

    wiki = load_dataset(
        "yuhuanstudio/wikipedia-zh",
        split="train",
        streaming=True
    )

    wiki = wiki.shuffle(
        seed=43,
        buffer_size=10000
    )


    train_path = (
        OUTPUT_DIR
        /
        "train.bin"
    )

    val_path = (
        OUTPUT_DIR
        /
        "val.bin"
    )


    # wb = write binary
    with open(
        train_path,
        "wb"
    ) as train_file, open(
        val_path,
        "wb"
    ) as val_file:


        # ==========================================
        # FineWeb2
        # ==========================================

        fineweb_stats = process_source(
            dataset=fineweb,
            source_name="fineweb2",
            tokenizer=tokenizer,

            train_target=(
                FINEWEB_TRAIN_TARGET
            ),

            val_target=(
                FINEWEB_VAL_TARGET
            ),

            train_file=train_file,
            val_file=val_file
        )


        # ==========================================
        # Wikipedia
        # ==========================================

        wiki_stats = process_source(
            dataset=wiki,
            source_name="wikipedia",
            tokenizer=tokenizer,

            train_target=(
                WIKI_TRAIN_TARGET
            ),

            val_target=(
                WIKI_VAL_TARGET
            ),

            train_file=train_file,
            val_file=val_file
        )


    # ========================================================
    # 统计最终结果
    # ========================================================

    train_tokens = (
        fineweb_stats["train_tokens"]
        +
        wiki_stats["train_tokens"]
    )

    val_tokens = (
        fineweb_stats["val_tokens"]
        +
        wiki_stats["val_tokens"]
    )


    metadata = {

        "tokenizer": TOKENIZER_PATH,

        "vocab_size": len(tokenizer),

        "dtype": "uint16",

        "seq_len": SEQ_LEN,

        "train_tokens": train_tokens,

        "val_tokens": val_tokens,

        "train_blocks": (
            (train_tokens - 1)
            //
            SEQ_LEN
        ),

        "val_blocks": (
            (val_tokens - 1)
            //
            SEQ_LEN
        ),

        "sources": {
            "fineweb2": fineweb_stats,
            "wikipedia": wiki_stats
        }
    }


    meta_path = (
        OUTPUT_DIR
        /
        "meta.json"
    )


    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )


    print("\n============================")
    print("Pretraining dataset finished")
    print("============================")

    print(
        f"Train tokens: {train_tokens:,}"
    )

    print(
        f"Val tokens: {val_tokens:,}"
    )

    print(
        f"Train blocks: "
        f"{metadata['train_blocks']:,}"
    )

    print(
        f"Val blocks: "
        f"{metadata['val_blocks']:,}"
    )

    print(
        "Saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()