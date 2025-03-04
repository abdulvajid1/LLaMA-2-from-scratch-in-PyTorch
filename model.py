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

    i = torch.arange(0, head_dim , 2)

    theta = 1 / ( theta ** ((i - 1) / head_dim))


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

    

