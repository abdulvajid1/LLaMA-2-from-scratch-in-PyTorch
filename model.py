import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class ModelArgs:

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = None
    norm_eps: float = 1e-5
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: float = None

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len : int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):

    assert head_dim % 2 == 0, "dimension should be divisible 2"

    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    i = torch.arange(0, head_dim , 2) # the -2 part will be already dene when taken the arange with step of 2

    #  the - from -2 is used as invers of the equation so 1/....
    theta = 1 / ( theta ** ((i - 1) / head_dim)) # we don't use -1 since we already starting from 0 value, we use -1 when i in the equation start from 1

    # construct the position m 
    m = torch.arange(0, seq_len, device=device)

    freq = torch.outer(m, theta).float()

    freq_complex = torch.polar(torch.ones_like(freq), freq)

    return freq_complex

def apply_rotary_postional_encoding(embedding: torch.Tensor, freq_complex_precomputed: torch.Tensor, device:str):
    embedding_complex = torch.view_as_complex(embedding.float().reshape(*embedding.shape[:-1], -1, 2))
    # add dimension to match with embedding dimension, we added dimension in batch and head, so it will broad cast
    freq_complex_precomputed= freq_complex_precomputed.unsqueeze(0).unsqueeze(2) # (seqlen, head/2) -> (1, seqlen, 1, head/2), it will broadcast with the embedding from model
    # Position wise multiplication
    embedding_rotated = embedding_complex * freq_complex_precomputed
    embedding_rotated_out = torch.view_as_real(embedding_rotated) # (batch, seq, head, head_dim/2, 2)
    # now we change this back to the original embedding shape
    embedding_out = embedding_rotated_out.reshape(*embedding)
    return embedding_out

# RMS Normalization Block
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6 ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.gamma_parameter = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps )
    
    def forward(self, x):
        return self.gamma_parameter * self._norm(x) # i have something to add here, if there is eror , check the code hre x.float().type_as(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freq_complex) # the vid used forward method directly here? why?
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class SelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q =  args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads_q)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim)
        self.wo = nn.Linear(self.head_dim * args.n_heads, args.dim)

        self.cache_k = torch.zeros((args.batch_size, args.max_seq_len, self.n_kv_heads , self.head_dim))
        self.cache_v = torch.zeros((args.batch_size, args.max_seq_len, self.n_kv_heads , self.head_dim))

    
    # number of heads in key and value can be differant from query heads since this is grouped query
    # we need to repeate the key metrix so it will match the same head number as query, so matrix multiplicatoin work without any problem
    def repeat_kv(self, x):
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        else:
            return ( 
                x[:,:,:,None,:].expand(batch_size, seq_len, self.n_kv_heads, self.n_rep ,self.head_dim)
                .reshape(batch_size, seq_len, n_kv_heads * self.n_rep , head_dim)
                )
    
    def forward(self, x, freq_complex, start_pos):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary postional embedding to query and key heads
        xq = apply_rotary_postional_encoding(xq, freq_complex)
        xk = apply_rotary_postional_encoding(xk, freq_complex)

        # add key and value matrix to cache
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # retrive all the key and value caches so for
        key =  self.cache_k[:batch_size, 0:start_pos+seq_len]
        value =  self.cache_k[:batch_size, 0:start_pos+seq_len]

        keys = self.repeat_kv(key)
        values = self.repeat_kv(value)

        # (B, 1, H, head_dim) -> (B, H, 1 head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # (B, H, 1, head_dim) * (B, H, head_dim, seq_len) -> (B, H, 1, Seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # (B, H, 1, Seq_len) * (B, H, Seq_len, head_dim) -> (B, H, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, 1, H, head_dim) -> (B, S, d_value)
        output = (output.transpose(1, 2).contiguous.view(batch_size, seq_len, -1))

        # return output with final linear layer to capture more complex pattern
        return self.wo(output)
            

# Transfomer Skelton
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embdding =  nn.Embedding(self.vocab_size,args.dim)

        self.layers = nn.ModuleList()

        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):

        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time can be processed"
        h = self.token_embdding(tokens)
        freq_complex = self.freq_complex[start_pos: start_pos + seq_len]

        for encoder_layer in self.layers:
            h = encoder_layer(h, start_pos, freq_complex)

    

