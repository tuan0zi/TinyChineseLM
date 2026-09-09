import torch

from transformers import AutoTokenizer

from model.config import ModelConfig
from model.model import TinyLlama


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=30
):

    model.eval()

    #推理的时候没有 dataset 和 dataLoader 所以要自己 把 文本转换成 tokenid （要求是整数），再打包成 batch
    # 1. 文本 → token ids
    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    # 2. 一个 token 一个 token 生成
    for _ in range(max_new_tokens):

        # 防止超过模型最大长度
        model_input = input_ids[
            :,
            -model.config.max_seq_len:
        ]

        # [B, T, V]
        logits = model(model_input)

        # 只取最后一个 token 位置
        # [B, V]
        next_token_logits = logits[:, -1, :]

        # 选择概率最大的 token
        # [B, 1]
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # 接到原序列后面
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        # 遇到 EOS 就停止
        if (
            next_token.item()
            ==
            tokenizer.eos_token_id
        ):
            break

    # token ids → 文本
    text = tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True
    )

    return text

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer/tokenizer_k",
        local_files_only=True
    )

    checkpoint = torch.load(
        "checkpoints/final.pt",
        map_location=device,
        weights_only=True
    )

    config = ModelConfig(
        **checkpoint["config"]
    )

    model = TinyLlama(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="中国的首都是",
        device=device,
        max_new_tokens=30
    )

    print(text)