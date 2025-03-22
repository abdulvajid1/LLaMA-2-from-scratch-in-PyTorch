from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class Llama:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, modelargs: ModelArgs):
        self.model = model 
        self.tokenizer = tokenizer
        self.model_args = modelargs

    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
            assert checkpoints > 0, "No checkpoints found"
            check_path = checkpoints[0]
            print(f'Loading checkpoint in {check_path}s')
            torch.load(check_path, map_location='cpu')
            print(f'Loading checkpoint in {(time.time() - prev_time):.2f}s')
            prev_time = time.time()
        with open(Path(checkpoint_dir) / "params.json","r") as f:
            params = json.loads(f.read())
        
        modelargs: ModelArgs = ModelArgs(max_batch_size=max_batch_size,
                                         max_seq_len=max_seq_len,
                                         **params)
            

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        modelargs.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Storage)

        model = Transformer(modelargs).to(device)

        if load_model:
            del checkpoints['rope.freq']
            model.load_state_dict(checkpoints, strict=True)
            print(f'Loading checkpoint in {(time.time() - prev_time):.2f}s')
        
        return Llama(model, tokenizer, modelargs)
    

if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'\
    
    model = Llama.build(
        checkpoint_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print('All Ok')


    # inferance



