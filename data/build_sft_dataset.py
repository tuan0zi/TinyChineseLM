import hashlib
import json
from pathlib import Path

from datasets import (
    get_dataset_config_names,
    load_dataset,
)


# ============================================================
# 1. 配置
# ============================================================

DATASET_NAME = "m-a-p/COIG-CQIA"

OUTPUT_DIR = Path("data/sft")

TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VAL_PATH = OUTPUT_DIR / "val.jsonl"
META_PATH = OUTPUT_DIR / "meta.json"


# 5% 用于 validation
VAL_RATIO = 0.05


# ============================================================
# 2. 文本清洗
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    return text.strip()


# ============================================================
# 3. 稳定地划分 Train / Validation
# ============================================================

def is_validation_sample(   # 是否将该条数据划分进验证集
    instruction,
    input_text,
    output
):

    # 把一个完整样本拼起来
    content = (
        instruction
        + "\0"
        + input_text
        + "\0"
        + output
    )

    # 根据内容计算固定 hash
    digest = hashlib.sha256(
        content.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big"
    )

    # 0 ~ 9999
    bucket = value % 10000

    return bucket < int(
        VAL_RATIO * 10000   
    )


# ============================================================
# 4. 主函数
# ============================================================

def main():

    #确保目录存在
    OUTPUT_DIR.mkdir(
        parents=True,  #表示 父目录 若不存在 也一起创建
        exist_ok=True  #若文件已经存在 不要报错
    )


    # 获取 COIG-CQIA 所有子数据集
    configs = get_dataset_config_names(
        DATASET_NAME
    )

    print(
        "Dataset configs:"
    )

    for config in configs:
        print(" -", config)


    train_count = 0
    val_count = 0

    duplicate_count = 0
    invalid_count = 0

    total_seen = 0


    # 用来去重
    seen = set()


    # 记录每个 subset 的统计
    source_stats = {}


    with open(
        TRAIN_PATH,
        "w",
        encoding="utf-8"
    ) as train_file, open(
        VAL_PATH,
        "w",
        encoding="utf-8"
    ) as val_file:


        # ====================================================
        # 遍历所有 subset
        # ====================================================

        for config_name in configs:

            print(
                f"\nLoading: {config_name}"
            )


            dataset = load_dataset(
                DATASET_NAME,
                config_name,
                split="train"
            )


            source_train = 0
            source_val = 0


            for item in dataset:

                total_seen += 1


                # --------------------------------------------
                # 读取三个核心字段
                # --------------------------------------------

                instruction = clean_text(
                    item.get(
                        "instruction",
                        ""
                    )  #该函数表示 如果有 就返回该值 如果没有就返回空字符串
                )

                input_text = clean_text(
                    item.get(
                        "input",
                        ""
                    )
                )

                output = clean_text(
                    item.get(
                        "output",
                        ""
                    )
                )


                # --------------------------------------------
                # 无 instruction / output 的样本不要  #input字段允许为空
                # --------------------------------------------

                if (
                    not instruction
                    or
                    not output
                ):

                    invalid_count += 1
                    continue


                # --------------------------------------------
                # 去重
                # --------------------------------------------

                sample_key = (
                    instruction,
                    input_text,
                    output
                )


                if sample_key in seen:

                    duplicate_count += 1
                    continue


                seen.add(
                    sample_key
                )


                # --------------------------------------------
                # 最终保存格式
                # --------------------------------------------

                sample = {

                    "instruction":
                        instruction,

                    "input":
                        input_text,

                    "output":
                        output,
                }


                line = (
                    json.dumps(
                        sample,
                        ensure_ascii=False
                    )
                    +
                    "\n"
                )


                # --------------------------------------------
                # 文档级 Train / Validation 划分
                # --------------------------------------------

                if is_validation_sample(
                    instruction,
                    input_text,
                    output
                ):

                    val_file.write(
                        line
                    )

                    val_count += 1
                    source_val += 1


                else:

                    train_file.write(
                        line
                    )

                    train_count += 1
                    source_train += 1


            source_stats[
                config_name
            ] = {

                "train_samples":
                    source_train,

                "val_samples":
                    source_val,
            }


            print(
                f"{config_name} finished | "
                f"train={source_train:,} | "
                f"val={source_val:,}"
            )


    # ========================================================
    # 5. 保存 Metadata
    # ========================================================

    metadata = {

        "dataset":
            DATASET_NAME,

        "train_samples":
            train_count,

        "val_samples":
            val_count,

        "total_samples":
            train_count + val_count,

        "duplicates_removed":
            duplicate_count,

        "invalid_removed":
            invalid_count,

        "val_ratio":
            VAL_RATIO,

        "sources":
            source_stats,
    }


    with open(
        META_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )


    # ========================================================
    # 6. 打印最终统计
    # ========================================================

    print(
        "\n"
        "============================"
    )

    print(
        "SFT dataset finished"
    )

    print(
        "============================"
    )

    print(
        f"Total seen: "
        f"{total_seen:,}"
    )

    print(
        f"Train samples: "
        f"{train_count:,}"
    )

    print(
        f"Val samples: "
        f"{val_count:,}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicate_count:,}"
    )

    print(
        f"Invalid removed: "
        f"{invalid_count:,}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()