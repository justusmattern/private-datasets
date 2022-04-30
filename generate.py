import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import sys
import argparse

def generate(model, tokenizer, prompts, num_sequences_per_prompt, filenames):

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer)
    model = GPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)
    model.parallelize()

    for prompt, num_sequences, file in zip(prompts, num_sequences_per_prompt, filenames):

        input_ids = tokenizer.encode(prompt + ' ', return_tensors='pt').to('cuda:0')
        final_samples = []

        while len(final_samples) < num_sequences:
            print('yeah')
            sample_outputs = model.generate(
                input_ids,
                do_sample=True, 
                max_length=512, 
                top_k=40, 
                top_p=0.95,
                no_repeat_ngram_size=3,
                num_return_sequences=10
            )

            
            for sample in sample_outputs:
                text = tokenizer.decode(sample, skip_special_tokens = False)
                print(text)
                text = text.split(tokenizer.eos_token)[0]
                final_samples.append(text)
        
        with open(file, 'w') as f:
            for sample in final_samples:
                f.write(sample.replace('\n', ' ')+'\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--tokenizer')
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--num-sequences-per-prompt', type=int, nargs='+')
    parser.add_argument('--filenames', type=str, nargs='+')

    args = parser.parse_args()

    generate(args.model, args.tokenizer, args.prompts, args.num_sequences_per_prompt, args.filenames)

