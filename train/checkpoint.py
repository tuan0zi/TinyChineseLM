import torch
from pathlib import Path
from dataclasses import asdict

def save_checkpoint(
    model,
    optimizer,
    config,
    epoch,
    global_step,
    save_path
):

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "config":
            asdict(config),

        "epoch":
            epoch,

        "global_step":
            global_step
    }

    torch.save(
        checkpoint,
        save_path
    )


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    epoch = checkpoint[
        "epoch"
    ]

    global_step = checkpoint[
        "global_step"
    ]

    return (
        epoch,
        global_step
    )