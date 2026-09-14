import torch

from transformers import AutoTokenizer

from model.config import ModelConfig
from model.model import TinyLlama


@torch.no_grad()
def chat_generate(
    model,
    tokenizer,
    user_text,
    device,
    max_new_tokens=128
):

    model.eval()


    # =========================
    # 1. 构造 ChatML Prompt
    # =========================

    prompt = (
        "<|im_start|>user\n"
        + user_text
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


    # =========================
    # 2. 文本 → Token IDs
    # =========================

    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )


    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )


    # 只记录 Assistant 新生成的 token
    generated_ids = []


    # =========================
    # 3. 自回归生成
    # =========================

    for _ in range(max_new_tokens):

        # 防止超过最大上下文长度
        model_input = input_ids[
            :,
            -model.config.max_seq_len:
        ]


        # [B, T, V]
        logits = model(
            model_input
        )


        # 最后一个位置
        # [B, V]
        next_token_logits = logits[
            :,
            -1,
            :
        ]


        # Greedy：
        # 每次选择概率最大的 token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )


        next_token_id = (
            next_token.item()
        )


        # =========================
        # 4. 遇到 <|im_end|> 停止
        # =========================

        if (
            next_token_id
            ==
            tokenizer.eos_token_id
        ):
            break


        generated_ids.append(
            next_token_id
        )


        # 接到上下文后面
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )


    # =========================
    # 5. Token IDs → Assistant文本
    # =========================

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )


    return answer


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # 注意：
    # 这里应该用你正式的 8K tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer/tokenizer_8k",
        local_files_only=True
    )


    # 注意：
    # 这里加载 SFT 后的模型
    checkpoint = torch.load(
        "checkpoints/sft_final.pt",
        map_location=device,
        weights_only=False
    )


    config = ModelConfig(
        **checkpoint["config"]
    )


    model = TinyLlama(
        config
    ).to(device)


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    answer = chat_generate(
        model=model,
        tokenizer=tokenizer,
        user_text="什么是机器学习？",
        device=device,
        max_new_tokens=128
    )


    print(
        "Assistant:",
        answer
    )