from multihead_attention import MultiHeadAttention
import torch
from torch import nn


embedding_layer = nn.Embedding(num_embeddings=64, embedding_dim=512)
indexes = torch.arange(0, 64, 1).unsqueeze(0) # Vector to be embedded must be within [0, embedding_dim - 1]

embedding = embedding_layer(indexes) # (1 x 64 x 512) embedding
d_emb = embedding.shape[-1]


mha = MultiHeadAttention(8, d_emb)
mha_out = mha(embedding)
print(mha_out.shape)

        