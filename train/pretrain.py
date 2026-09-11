import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.packed_pretrain_dataset import PackedPretrainDataset
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
    # 3. Dataset
    # =========================
    train_dataset = PackedPretrainDataset(
        data_path="data/packed/train.bin",
        seq_len=512
    )

    val_dataset = PackedPretrainDataset(
        data_path="data/packed/val.bin",
        seq_len=512
    )



    # =========================
    # 4. DataLoader
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

    # =========================
    # 5. 模型配置
    # =========================
    config = ModelConfig(
        vocab_size=len(tokenizer),
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        intermediate_dim=344,
        max_seq_len=512
    )


    # =========================
    # 6. 模型
    # =========================
    model = TinyLlama(config)

    model = model.to(device)

    model.train()


    # =========================
    # 7. 优化器
    # =========================
    max_lr = 3e-4
    min_lr = 3e-5


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=0.01
    )

    # =========================
    # 8. 训练
    # =========================
    num_epochs = 3
    total_steps = (
        len(train_dataloader)
        *
        num_epochs
    )


    warmup_steps = max(
        1,
        int(
            total_steps * 0.1
        )
    )

    global_step = 0


    for epoch in range(num_epochs):   #和overfit 不同的是 加上了 epoch 同时在整个dataloader 上训练 不只是在一个batch 上学习

        for batch in train_dataloader:

            input_ids = batch["input_ids"].to(device)

            labels = batch["labels"].to(device)

            # 计算当前学习率
            current_lr = get_lr(
                step=global_step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                max_lr=max_lr,
                min_lr=min_lr
            )


            # 设置当前学习率
            for param_group in optimizer.param_groups:

                param_group["lr"] = current_lr

            optimizer.zero_grad()


            logits = model(
                input_ids
            )


            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    config.vocab_size
                ),
                labels.reshape(-1),
                ignore_index=-100
            )


            loss.backward()

            torch.nn.utils.clip_grad_norm_(  
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()


            global_step += 1

            if global_step % 500 == 0:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    save_path=(
                        f"checkpoints/"
                        f"step_{global_step}.pt"
                    )
                )

                print(
                    f"checkpoint saved: "
                    f"step {global_step}"
                )
            
            print(
                f"epoch {epoch + 1} | "
                f"step {global_step} | "
                f"loss {loss.item():.4f}"
            )
        
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

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=num_epochs - 1,
        global_step=global_step,
        save_path="checkpoints/final.pt"
    )        


if __name__ == "__main__":
    train()