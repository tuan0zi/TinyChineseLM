import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    vocab_size
):

    model.eval()

    total_loss = 0.0
    num_batches = 0


    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)


        logits = model(
            input_ids
        )


        loss = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size
            ),
            labels.reshape(-1),
            ignore_index=-100
        )


        total_loss += loss.item()

        num_batches += 1


    avg_loss = (
        total_loss
        /
        num_batches
    )


    ppl = math.exp(
        avg_loss
    )


    model.train()


    return avg_loss, ppl