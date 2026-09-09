import math


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float
):

    # ========================
    # 1. Warmup
    # ========================

    if step < warmup_steps:

        lr = (
            max_lr
            *
            (step + 1)
            /
            warmup_steps
        )

        return lr


    # ========================
    # 2. Cosine Decay
    # ========================

    decay_steps = max(
        1,
        total_steps
        -
        warmup_steps
        -
        1
    )


    progress = (
        step - warmup_steps
    ) / decay_steps


    progress = min(
        progress,
        1.0
    )


    cosine = 0.5 * (
        1.0
        +
        math.cos(
            math.pi * progress
        )
    )


    lr = (
        min_lr
        +
        cosine
        *
        (
            max_lr
            -
            min_lr
        )
    )


    return lr