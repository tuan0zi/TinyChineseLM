import json
import os
from typing import Generator

from transformers import AutoTokenizer
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
)
from tokenizers.normalizers import NFKC


# ============================================================
# 1. 读取 JSONL 训练数据
# ============================================================

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """
    逐行读取 JSONL 文件，并提取每一行中的 text 字段。

    假设数据格式如下：
    {"text": "我正在学习大语言模型。"}
    {"text": "BPE 是一种常见的分词算法。"}
    """

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                if "text" not in data:
                    raise KeyError(   #主动抛出异常
                        f"Missing 'text' field in line {line_num}"
                    )

                yield data["text"]

            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue

            except KeyError as e:
                print(e)
                continue


# ============================================================
# 2. 创建 Transformers 需要的 Tokenizer 配置文件
# ============================================================

def create_tokenizer_config(save_dir: str) -> None:
    """
    创建：
    1. tokenizer_config.json
    2. special_tokens_map.json
    """

    config = {
        "add_bos_token": False,
        "add_eos_token": False,

        # 和 ByteLevel(add_prefix_space=False) 保持一致
        "add_prefix_space": False,

        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",

        # 这里是一个很大的占位值，并不代表模型真的支持这么长的上下文
        "model_max_length": 1000000000000000019884624838656,

        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",

        # Chat Template
        "chat_template": (
            "{% for message in messages %}"

            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n"
            "{{ message['content'] }}"
            "<|im_end|>\n"

            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n"
            "{{ message['content'] }}"
            "<|im_end|>\n"

            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n"
            "{{ message['content'] }}"
            "<|im_end|>\n"

            "{% endif %}"
            "{% endfor %}"

            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        ),
    }

    # 保存 tokenizer_config.json
    tokenizer_config_path = os.path.join(
        save_dir,
        "tokenizer_config.json"
    )

    with open(
        tokenizer_config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4  #每级缩进4个空格
        )

    # 特殊 Token 映射
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",

        "additional_special_tokens": [
            "<s>",
            "</s>"  #兼容一些旧格式或其他模型格式
        ]
    }

    special_tokens_map_path = os.path.join(
        save_dir,
        "special_tokens_map.json"
    )

    with open(
        special_tokens_map_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            special_tokens_map,
            f,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 3. 训练 BPE Tokenizer
# ============================================================

def train_tokenizer(
    data_path: str,
    save_dir: str,
    vocab_size: int = 8192
) -> None:

    """
    使用 JSONL 文本训练 ByteLevel BPE Tokenizer，
    并保存训练结果。
    """

    # 如果保存目录不存在，则创建
    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 初始化 BPE Tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>"
        )
    )

    # Unicode 规范化
    tokenizer.normalizer = NFKC()

    # ByteLevel 预分词
    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )
                                                 #这几个的左边都是Tokenizer对象的属性 右边是各自类的实例化
    # ByteLevel 解码器
    tokenizer.decoder = (
        decoders.ByteLevel()
    )

    # --------------------------------------------------------
    # 特殊 Token
    # --------------------------------------------------------

    special_tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    # --------------------------------------------------------
    # BPE Trainer
    # --------------------------------------------------------

    trainer = trainers.BpeTrainer(

        # 最终词表目标大小
        vocab_size=vocab_size,

        # 特殊 Token
        special_tokens=special_tokens,

        # pair 至少出现 2 次才考虑 merge
        min_frequency=2,

        # 显示训练进度
        show_progress=True,

        # ByteLevel 的 256 个基本 byte 符号
        initial_alphabet=(
            pre_tokenizers.ByteLevel.alphabet()
        ),
    )

    print(
        f"Training tokenizer with data from: {data_path}"
    )

    # --------------------------------------------------------
    # 统计 JSONL 行数
    # --------------------------------------------------------

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as f:
        num_lines = sum(1 for _ in f)

    print(
        f"Number of training samples: {num_lines}"
    )

    # --------------------------------------------------------
    # 创建文本 Generator
    # --------------------------------------------------------

    texts = read_texts_from_jsonl(
        data_path
    )

    # --------------------------------------------------------
    # 真正开始训练 BPE
    # --------------------------------------------------------

    tokenizer.train_from_iterator(   #真正启动训练的函数
        texts,
        trainer=trainer,
        length=num_lines
    )

    # --------------------------------------------------------
    # 验证特殊 Token ID
    # --------------------------------------------------------

    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4

    except AssertionError as e:
        print(
            "Special tokens mapping error:",
            e
        )
        raise

    print("\n=== Special Token Mapping ===")
    print(
        "<unk>         ->",
        tokenizer.token_to_id("<unk>")
    )
    print(
        "<s>           ->",
        tokenizer.token_to_id("<s>")
    )
    print(
        "</s>          ->",
        tokenizer.token_to_id("</s>")
    )
    print(
        "<|im_start|>  ->",
        tokenizer.token_to_id("<|im_start|>")
    )
    print(
        "<|im_end|>    ->",
        tokenizer.token_to_id("<|im_end|>")
    )

    # --------------------------------------------------------
    # 保存 tokenizer.json
    # --------------------------------------------------------

    tokenizer_path = os.path.join(
        save_dir,
        "tokenizer.json"
    )

    tokenizer.save(
        tokenizer_path
    )

    # --------------------------------------------------------
    # 保存 Transformers 配置
    # --------------------------------------------------------

    create_tokenizer_config(
        save_dir
    )

    print(
        f"\nTokenizer saved to: {save_dir}"
    )


# ============================================================
# 4. 测试训练完成的 Tokenizer
# ============================================================

def eval_tokenizer(
    tokenizer_path: str
) -> None:

    """
    对训练完成的 Tokenizer 进行简单测试。
    """

    # --------------------------------------------------------
    # 加载 Tokenizer
    # --------------------------------------------------------

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True   #只允许在本地上读取
        )

    except Exception as e:
        print(
            f"Error loading tokenizer: {e}"
        )
        return

    # --------------------------------------------------------
    # Tokenizer 基本信息
    # --------------------------------------------------------

    print(
        "\n=== Tokenizer 基本信息 ==="
    )

    print(
        f"Vocab size: {len(tokenizer)}"
    )

    print(
        f"Special tokens: "
        f"{tokenizer.all_special_tokens}"
    )

    print(
        f"Special token IDs: "
        f"{tokenizer.all_special_ids}"
    )

    # --------------------------------------------------------
    # 测试 Chat Template
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": "你是一个AI助手。"
        },
        {
            "role": "user",
            "content": "How are you?"
        },
        {
            "role": "assistant",
            "content":
                "I'm fine, thank you. and you?"
        },
        {
            "role": "user",
            "content": "I'm good too."
        },
        {
            "role": "assistant",
            "content":
                "That's great to hear!"
        },
    ]

    print(
        "\n=== 聊天模板测试 ==="
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,

        # 如果是让模型接下来继续回答，
        # 可以改为 True
        add_generation_prompt=False
    )

    print(
        "Generated prompt:\n",
        prompt,
        sep=""    #print(A,B) sep是A与B之间的填充
    )

    # --------------------------------------------------------
    # 测试编码 / 解码
    # --------------------------------------------------------

    print(
        "\n=== 编码解码测试 ==="
    )

    encoded = tokenizer(
        prompt,
        truncation=True,  #如果分词后的 Token 数量太长，允许截断。
        max_length=256
    )

    print(
        "Input IDs:"
    )

    print(
        encoded["input_ids"]
    )

    # 把 ID 转换成 Token，方便观察
    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"]
    )

    print(
        "\nTokens:"
    )

    print(
        tokens
    )

    decoded = tokenizer.decode( #Token ID 查回词表中的 Token，再由对应的 Decoder（你这里是 ByteLevel Decoder）把 Token 内部保存的 byte 表示还原成原始字节，最后按 UTF-8 重新得到文本。它不是神经网络生成，也不是通过语义猜原文，而是确定性的反向映射。
        encoded["input_ids"],
        skip_special_tokens=False
    )

    print(
        "\nDecoded text:"
    )

    print(
        decoded
    )

    print(
        "\nDecoded text matches original:",
        decoded == prompt
    )

    # --------------------------------------------------------
    # 测试特殊 Token
    # --------------------------------------------------------

    print(
        "\n=== 特殊 Token 处理 ==="
    )

    test_text = (
        "<|im_start|>user\n"
        "Hello"
        "<|im_end|>"
    )

    encoded_special = tokenizer(
        test_text
    ).input_ids

    print(
        "Encoded IDs:",
        encoded_special
    )

    print(
        "Encoded Tokens:",
        tokenizer.convert_ids_to_tokens(
            encoded_special
        )
    )

    decoded_special = tokenizer.decode(
        encoded_special,
        skip_special_tokens=False
    )

    print(
        f"Original: {test_text}"
    )

    print(
        f"Decoded: {decoded_special}"
    )

    print(
        "Special tokens preserved:",
        decoded_special == test_text
    )

    # --------------------------------------------------------
    # 普通文本 BPE 测试
    # --------------------------------------------------------

    print(
        "\n=== BPE 分词测试 ==="
    )

    test_texts = [
        "unhappiness",
        "newest",
        "learning",
        "我正在学习大语言模型。",
        "训练 tokenizer 不需要 GPU。",
    ]

    for text in test_texts:

        encoded = tokenizer(
            text,
            add_special_tokens=False
        )

        tokens = tokenizer.convert_ids_to_tokens(
            encoded["input_ids"]
        )

        print(
            f"\nText: {text}"
        )

        print(
            f"Tokens: {tokens}"
        )

        print(
            f"IDs: {encoded['input_ids']}"
        )


# ============================================================
# 5. 主函数
# ============================================================

def main():

    # --------------------------------------------------------
    # 修改成你的训练数据路径
    # --------------------------------------------------------

    data_path = "data/tokenizer/tokenizer_train.jsonl"

    # Tokenizer 保存路径
    save_dir = "tokenizer/tokenizer_8k"

    # --------------------------------------------------------
    # 训练 Tokenizer
    # --------------------------------------------------------

    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=8192
    )

    # --------------------------------------------------------
    # 评估 Tokenizer
    # --------------------------------------------------------

    eval_tokenizer(
        save_dir
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()