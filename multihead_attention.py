import torch
from torch import nn
import self_attention
from self_attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, in_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.n_heads = n_heads
        
        assert self.in_dim % self.n_heads == 0, "Embedding dimension is not divisible to number of heads"
        self.attn_dim = self.in_dim // self.n_heads
        
        self.heads = nn.ModuleList(
            [Attention(self.attn_dim, self.attn_dim) for _ in range(self.n_heads)])
        
        self.linear = nn.Linear(self.in_dim , self.in_dim)
    def forward(self, embedding, mask=None):

        batch, seq_len, d_emb = embedding.shape
        
         # (1 x 64 x 512) --> (1 x 64 x 8 x 64) which means : 
        # (Batch, seq_len,d_emb) -->(Batch, seq_len, heads, attn_dim)  
        mha_input = embedding.view(batch, seq_len, self.n_heads, self.attn_dim)

        # (batch, seq_len, n_heads, attn_dim) -> (batch, n_heads, seq_len, attn_dim)    
        mha_input = mha_input.permute(0, 2, 1 ,3) # it divides the embedding to heads
        outs = [head(mha_input[:, i], mask=None) for i, head in enumerate(self.heads) ]
        out = torch.cat(outs, dim=-1)
        
        
        out = self.linear(out)
        
        return out


embedding_layer = nn.Embedding(num_embeddings=64, embedding_dim=512)
indexes = torch.arange(0, 64, 1).unsqueeze(0) # Vector to be embedded must be within [0, embedding_dim - 1]

embedding = embedding_layer(indexes) # (1 x 64 x 512) embedding
d_emb = embedding.shape[-1]


mha = MultiHeadAttention(8, d_emb)
mha_out = mha(embedding)
print(mha_out.shape)

        