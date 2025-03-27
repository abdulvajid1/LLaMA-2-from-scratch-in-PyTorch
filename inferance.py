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
    
    def text_completion(self, prompts:list[str], temperature: float=0.6,top_p: float=0.9, max_gen_len:Optional[int]=None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1

        # Tokenize the multiple prompts at once 
        prompts_tokens =[self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=True) for prompt in prompts]

        # make sure batch_size in not over size
        batch_size = len(prompts_tokens) # consider the number of prompts at once as the batch size
        assert batch_size <= self.model_args.max_batch_size

        # make sure the prompt size is not over the limit
        max_prompt_len = max((len(prompt)) for prompt in prompts_tokens)
        assert max_prompt_len <= self.model_args.max_seq_len

        total_len = min(self.model_args.max_seq_len, max_prompt_len+max_gen_len )

        # Initialize a Tensor with max size of the output token with batchsize full of pad_token
        pad_id = self.tokenizer.pad_id()
        token = torch.full((batch_size, total_len),
                           fill_value=pad_id,
                            dtype=torch.long,
                            device=device)
        
        # Initialize at the start the prompt token for each batch tokens
        for k, t in enumerate(prompts_tokens):
            token[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Initialize the end of token  for each batch to identify any of the batch get end token init
        eos_reached = torch.Tensor([False]*batch_size, device=device)

        prompt_token_mask = token != pad_id

        for cur_pos in tqdm(range(1, total_len), desc='Genarating next token'):
            with torch.no_grad():
                logits = self.model.forward(token[:, cur_pos-1: cur_pos],cur_pos)

            if temperature > 0:
                # The temperature is applied Before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy token
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

    
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_token_mask[:, cur_pos], token[:, cur_pos], next_token)
            token[:, cur_pos] = next_token

            # check if any of batch's token is not a prompt token which then will be a pad token and the next token is eos token
            # then for the current batch we update the eos_reached token as True
            eos_reached = eos_reached | (~prompt_token_mask[:, cur_pos]) & (next_token==self.tokenizer.eos_id())
            
            # check if all prompts (batch) got eos_token, stop genaration
            if all(eos_reached):
                break

            out_token = []
            out_text = []

            # itrate through each prompts output
            for prompt_token, curr_prompt_tokens in enumerate(token.tolist()):
                if self.tokenizer.eos_id() in curr_prompt_tokens:
                    eos_index = curr_prompt_tokens.index(self.tokenizer.eos_id())
                    curr_prompt_tokens = curr_prompt_tokens[:eos_index]
                out_token.append(curr_prompt_tokens)
                out_text.append(self.tokenizer.decode(curr_prompt_tokens))
            return (out_text, out_token)
        
        def _sample_top_p(self, probs, p):
            probs, prob_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum_prob = torch.cumsum(probs,dim=-1)
            p_mask = cumsum_prob - probs > p
            probs[p_mask] = 0 
            probs = probs.div(probs.sum(dim=-1,keepdim=True))
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(prob_idx, -1, next_token)

            return next_token








if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = ('cuda' if torch.cuda.is_available() and allow_cuda else 'cpu')
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


