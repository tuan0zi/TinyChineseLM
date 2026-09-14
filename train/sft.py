import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.sft_dataset import SFTDataset
from model.config import ModelConfig
from model.model import TinyLlama
from train.scheduler import get_lr
from train.checkpoint import save_checkpoint
from eval.evaluate import evaluate


def train():

    # =========================
    # 1. 设备
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)


    # =========================
    # 2. Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer/tokenizer_8k",
        local_files_only=True
    )


    # =========================
    # 3. 加载预训练 checkpoint
    # =========================
    checkpoint = torch.load(
        "checkpoints/final.pt",
        map_location="cpu",
        weights_only=False
    )


    # checkpoint 中的 config
    # 可能是 ModelConfig，也可能是 dict
    saved_config = checkpoint["config"]

    if isinstance(saved_config, dict):

        config = ModelConfig(
            **saved_config
        )

    else:

        config = saved_config


    print("model config:", config)


    # =========================
    # 4. Dataset
    # =========================
    train_dataset = SFTDataset(
        data_path="data/sft/train.jsonl",
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )

    val_dataset = SFTDataset(
        data_path="data/sft/val.jsonl",
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )


    # =========================
    # 5. DataLoader
    # =========================
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )


    print(
        "train samples:",
        len(train_dataset)
    )

    print(
        "val samples:",
        len(val_dataset)
    )


    # =========================
    # 6. 模型
    # =========================
    model = TinyLlama(config)


    # 关键：
    # 加载 Pretraining 学到的参数
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(device)

    model.train()


    print(
        "pretrained model loaded"
    )


    # =========================
    # 7. 优化器
    # =========================

    # SFT 使用更小的学习率
    max_lr = 2e-5
    min_lr = 2e-6


    # 不加载预训练 optimizer
    # SFT 重新创建一个 optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=0.01
    )


    # =========================
    # 8. 训练参数
    # =========================
    num_epochs = 2

    total_steps = (
        len(train_dataloader)
        *
        num_epochs
    )


    # SFT warmup 少一点
    warmup_steps = max(
        1,
        int(
            total_steps * 0.05
        )
    )


    global_step = 0


    # =========================
    # 9. 正式 SFT
    # =========================
    for epoch in range(num_epochs):

        # evaluate() 之后模型可能处于 eval 模式
        # 所以每个 epoch 开始重新设为 train
        model.train()


        for batch in train_dataloader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )


            # =====================
            # 当前学习率
            # =====================
            current_lr = get_lr(
                step=global_step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                max_lr=max_lr,
                min_lr=min_lr
            )


            for param_group in optimizer.param_groups:

                param_group["lr"] = (
                    current_lr
                )


            # =====================
            # 清空梯度
            # =====================
            optimizer.zero_grad()


            # =====================
            # Forward
            # =====================
            logits = model(
                input_ids
            )


            # =====================
            # SFT Loss
            # =====================
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    config.vocab_size
                ),
                labels.reshape(-1),
                ignore_index=-100
            )


            # =====================
            # Backward
            # =====================
            loss.backward()


            # =====================
            # Gradient Clipping
            # =====================
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )


            # =====================
            # 更新参数
            # =====================
            optimizer.step()


            global_step += 1


            # =====================
            # 保存中间 checkpoint
            # =====================
            if global_step % 500 == 0:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    save_path=(
                        f"checkpoints/"
                        f"sft_step_{global_step}.pt"
                    )
                )


                print(
                    f"checkpoint saved: "
                    f"step {global_step}"
                )


            # =====================
            # 打印训练信息
            # =====================
            print(
                f"epoch {epoch + 1} | "
                f"step {global_step} | "
                f"lr {current_lr:.6e} | "
                f"loss {loss.item():.4f}"
            )


        # =========================
        # 10. Validation
        # =========================
        val_loss, val_ppl = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            vocab_size=config.vocab_size
        )


        print(
            f"epoch {epoch + 1} | "
            f"val_loss {val_loss:.4f} | "
            f"ppl {val_ppl:.2f}"
        )


    # =========================
    # 11. 保存最终 SFT 模型
    # =========================
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=num_epochs - 1,
        global_step=global_step,
        save_path="checkpoints/sft_final.pt"
    )


    print(
        "SFT finished | "
        "saved to checkpoints/sft_final.pt"
    )


if __name__ == "__main__":
    train()