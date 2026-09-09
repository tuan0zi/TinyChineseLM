import json

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.data = []

        with open(
            data_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                item = json.loads(line)

                self.data.append(
                    item["text"]
                )


    def __len__(self):

        return len(self.data)


    def __getitem__(
        self,
        index
    ):

        text = self.data[index]


        # 文本 -> token ids
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )


        # 最多保留 max_length 个正文 token
        ids = ids[:self.max_length]


        # 手动添加 EOS
        ids.append(
            self.tokenizer.eos_token_id
        )


        # 记录真实 token 数量
        valid_length = len(ids)


        # padding 到 max_length + 1
        pad_length = (
            self.max_length
            + 1
            - len(ids)
        )

        ids = (
            ids
            +
            [self.tokenizer.pad_token_id]
            * pad_length
        )


        ids = torch.tensor(
            ids,
            dtype=torch.long
        )


        # 输入
        input_ids = ids[:-1]

        # 下一个 token 作为标签
        labels = ids[1:].clone()


        # 有效 label 数量
        valid_label_length = (
            valid_length - 1
        )


        # padding 区域不计算 loss
        labels[
            valid_label_length:
        ] = -100


        return {
            "input_ids": input_ids,
            "labels": labels
        }