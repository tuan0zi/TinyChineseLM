import json

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []

        with open(
            data_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                self.data.append(item)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        item = self.data[index]

        instruction = item["instruction"]

        input_text = item.get(
            "input",
            ""
        )

        output = item["output"]


        # =====================================
        # 1. 构造 User 内容
        # =====================================

        if input_text:

            user_content = (
                instruction
                + "\n"
                + input_text
            )

        else:

            user_content = instruction


        # =====================================
        # 2. 构造 ChatML
        # =====================================

        prompt = (
            "<|im_start|>user\n"
            + user_content
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )

        answer = (
            output
            + "<|im_end|>"
        )


        # =====================================
        # 3. Tokenize
        # =====================================

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False
        )


        # =====================================
        # 4. 截断
        # =====================================

        # 后面需要 shift，
        # 所以最多准备 max_length + 1 个 token

        max_total_length = (
            self.max_length + 1
        )


        # 最多让 prompt 占一半，
        # 给 assistant 回答留出空间
        max_prompt_length = (
            max_total_length // 2
        )

        prompt_ids = prompt_ids[
            :max_prompt_length
        ]


        remaining_length = (
            max_total_length
            -
            len(prompt_ids)
        )

        answer_ids = answer_ids[
            :remaining_length
        ]


        # =====================================
        # 5. 拼接
        # =====================================

        full_ids = (
            prompt_ids
            +
            answer_ids
        )


        # User / Prompt 不计算 loss
        full_labels = (
            [-100] * len(prompt_ids)
            +
            answer_ids
        )


        # =====================================
        # 6. Next Token Prediction shift
        # =====================================

        input_ids = full_ids[:-1]

        labels = full_labels[1:]


        # =====================================
        # 7. Padding
        # =====================================

        pad_length = (
            self.max_length
            -
            len(input_ids)
        )


        if pad_length > 0:

            input_ids += (
                [self.tokenizer.pad_token_id]
                *
                pad_length
            )

            labels += (
                [-100]
                *
                pad_length
            )


        return {

            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }