import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import sys
import argparse
from utils import write_to_file

def generate(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompts, num_sequences_per_prompt, filenames, typical_decoding, return_texts):

    model = model
    tokenizer = tokenizer
    model.parallelize()
    model.eval()
    sample_collections = []
    for prompt, num_sequences, file in zip(prompts, num_sequences_per_prompt, filenames):
        

        input_ids = tokenizer.encode(prompt + ' ', return_tensors='pt').to('cuda:0')
        final_samples = []

        while len(final_samples) < num_sequences:

            if typical_decoding:
                sample_outputs = model.sample(
                    input_ids,
                    max_length=300,
                    typical_p=0.1,
                    no_repeat_ngram_size=3,
                    num_return_sequences=1
                )

            else:
                sample_outputs = model.generate(
                    input_ids,
                    do_sample=True, 
                    max_length=512,  
                    top_p=0.8,
                    no_repeat_ngram_size=3,
                    num_return_sequences=10
                )

            
            for sample in sample_outputs:
                text = tokenizer.decode(sample, skip_special_tokens = False)
                print(text)
                text = text.split(tokenizer.eos_token)[0]
                final_samples.append(text)

        if not return_texts:
            write_to_file(final_samples, file)
        
        sample_collections.append(final_samples)

    if return_texts:

        return sample_collections
            
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--tokenizer')
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--num-sequences-per-prompt', type=int, nargs='+')
    parser.add_argument('--filenames', type=str, nargs='+')
    parser.add_argument('--typical-decoding', action='store_true')

    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    model = GPT2LMHeadModel.from_pretrained(args.model, pad_token_id=tokenizer.eos_token_id)


    generate(model, tokenizer, args.prompts, args.num_sequences_per_prompt, args.filenames, args.typical_decoding, return_texts=False)

