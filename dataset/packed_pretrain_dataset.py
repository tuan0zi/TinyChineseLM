import numpy as np
import torch

from torch.utils.data import Dataset


class PackedPretrainDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        seq_len: int = 512
    ):

        self.seq_len = seq_len

        # 不把整个文件一次性读进内存
        # 而是建立磁盘文件到内存地址的映射
        self.data = np.memmap(
            data_path,
            dtype=np.uint16,
            mode="r"
        )


    def __len__(self):

        return (
            len(self.data) - 1
        ) // self.seq_len


    def __getitem__(self, index):

        start = (
            index
            *
            self.seq_len
        )

        end = (
            start
            +
            self.seq_len
            +
            1
        )


        chunk = self.data[
            start:end
        ]


        # numpy uint16
        # →
        # torch.long
        ids = torch.tensor(
            chunk.astype(np.int64),
            dtype=torch.long
        )


        input_ids = ids[:-1]

        labels = ids[1:]


        return {
            "input_ids": input_ids,
            "labels": labels
        }