import torch

from embedding import TokenEmbedding



vocab_size = 1000

hidden_dim = 128


embedding = TokenEmbedding(
    vocab_size,
    hidden_dim
)


input_ids = torch.tensor(
    [
        [1,5,10],
        [2,6,20]
    ]
)


output = embedding(
    input_ids
)


print(output.shape)