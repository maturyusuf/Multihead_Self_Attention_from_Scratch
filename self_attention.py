import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        
        self.Q_linear = nn.Linear(in_dim, out_dim)
        self.K_linear = nn.Linear(in_dim, out_dim)
        self.V_linear = nn.Linear(in_dim, out_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input_embedding, mask):
        
        q = self.Q_linear(input_embedding)
        k = self.K_linear(input_embedding)
        v = self.V_linear(input_embedding)
        
        d_k = q.shape[-1]
        
        # (1, 64 x 512) * (1, 512 x 64) -> (1, 64 x 64)
        # (1, 64 x 64) * (1, 64 x 512) -> (1, 64 x 512)
        
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, 1e-6)

        probability_matrix = self.softmax(attention_score)
        
        output = probability_matrix @ v
            
        return output
    


        
        
        
        