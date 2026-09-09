import json
import random
import os

random.seed(42)

texts = [
    "我正在学习大语言模型。",
    "大语言模型可以理解和生成自然语言。",
    "Transformer 是现代大语言模型的重要基础。",
    "Tokenizer 可以把文本转换成 token。",
    "BPE 是一种常见的子词分词算法。",
    "机器学习和深度学习非常有趣。",
    "我正在学习 Python 和 PyTorch。",
    "人工智能正在快速发展。",
    "今天我们训练一个自己的 tokenizer。",
    "训练 tokenizer 不需要使用 GPU。",
    "happy happiness unhappy unhappiness",
    "new newer newest",
    "play player playing played",
    "learn learner learning learned",
    "language model large language model",
]

os.makedirs("data", exist_ok=True)

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for _ in range(5000):
        text = random.choice(texts)

        data = {
            "text": text
        }

        f.write(
            json.dumps(data, ensure_ascii=False)
            + "\n"
        )

print("数据集生成完成：data/train.jsonl")