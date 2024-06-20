# Text Encoder: Transformer
# 63M parameter 12-layer, 512 wide model with 8-attention head
# BPE representation 49152 vocab size
# max sequence: 76
# start: [SOS], end: [EOS] 
# highest layer of the transformer at [EOS]: feature representation
# -> layer normalized and linearly projected into the multi-modal embedding space
# Masked self-attention was used in the text encoder
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import math
import numpy as np

class ResidualAttentionBlock(nn.Module): #Transformer encoder
    def __init__(self,model_dim:int, n_heads:int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(model_dim,n_heads)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",nn.Linear(model_dim,model_dim*4)),
            ("gelu",nn.GELU()),
            ("c_proj",nn.Linear(model_dim*4,model_dim))
        ]))
        self.ln_2 = nn.LayerNorm(model_dim)
        self.attn_mask = attn_mask
    
    def attention(self,x:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype = x.dytpe) if self.attn_mask is not None else None
        return self.attn(x,x,x,attn_mask = self.attn_mask)[0]
    
    def forward(self,x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class positional_encoding(nn.Module):
    def __init__(self,n_seq,model_dim):
        super().__init__()
        self.pos_emb = torch.empty(n_seq,model_dim)
        number = torch.arange(0,n_seq,dtype=torch.float).view(-1,1)
        division_term = torch.exp(math.log(10000)*(torch.arange(0,model_dim,2)/model_dim))
        self.pos_emb.data[:,0::2] = torch.sin(number/division_term)
        self.pos_emb.data[:,1::2] = torch.cos(number/division_term)
    
        self.pos_emb = self.pos_emb.unsqueeze(0).transpose(0,1)
    
    def forward(self,x):
        x = x + self.pos_emb
        return x

    

class Transformer(nn.Module):
    def __init__(self,max_length,vocab_size,n_layers,model_dim,attn_heads,attn_mask):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,model_dim)
        self.position_embedding = positional_encoding(max_length,model_dim)
        
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(model_dim,attn_heads,attn_mask) for _ in range(n_layers)])
    
    def forward(self,x):
        x = self.token_embedding(x)
        x = self.position_embedding(x)
        x = self.resblocks(x)

        return x

