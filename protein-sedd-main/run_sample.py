import torch
import argparse

#from data import CharacterTokenizer
from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling



def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    parser.add_argument("--dataset", default="acyp", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model(args.model_path, device)
    #tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast(
        vocab_file='vocab.json',
        merges_file='merges.txt',
        #bos_token='<s>',
        #eos_token='</s>',
        #unk_token='<unk>',
        #pad_token='<pad>',
        #mask_token='<mask>'
    )

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    )

    samples = sampling_fn(model)

    text_samples = tokenizer.batch_decode(samples)
    for i in text_samples:
        print(i[:25])
        print("=================================================")




if __name__=="__main__":
    main()
