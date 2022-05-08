import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import optim
from torch import nn
from utils import get_data_from_txt, pre_process, prepare_training_data, write_to_file
from text_dataset import Dataset
from train_lm import train_lm
from generate import generate

def run_pipeline(base_model: str, base_tokenizer: str, training_file_path: str, train_epochs: int, prompts: List[str], train_batch_size: int, train_mismatch_loss: bool, 
                train_mismatch_weight: float, model_out_file: str, generated_filenames: List[str], typical_decoding: bool, dp_optimization: bool, num_sequences_per_prompt: List[int], epsilon: float):

    train_data, train_loader = prepare_training_data(training_file_path, train_batch_size)

    trained_model = train_lm(
        args_model=base_model,
        args_tokenizer=base_tokenizer, 
        args_epochs=train_epochs,
        args_prompt =prompts,
        args_batch_size=train_batch_size, 
        args_mismatch_loss=train_mismatch_loss,
        args_mismatch_weight=train_mismatch_weight,
        args_model_out=model_out_file,
        return_resuls=True,
        train_data=train_data,
        train_loader=train_loader,
        args_dp_optimization=dp_optimization,
        epsilon=epsilon
    )

    sample_collection = generate(
        model=trained_model,
        tokenizer=GPT2Tokenizer.from_pretrained(base_tokenizer),
        prompts=prompts,
        num_sequences_per_prompt=num_sequences_per_prompt,
        filenames=generated_filenames,
        typical_decoding=typical_decoding,
        return_texts=True
    )

    # later: add code to clean texts

    for file, samples in zip(generated_filenames, sample_collection):
        write_to_file(samples, file)


def run(args):
    run_pipeline(
        base_model=args.base_model, 
        base_tokenizer=args.base_tokenizer, 
        training_file_path=args.training_file_path, 
        train_epochs=args.train_epochs, 
        prompts=args.prompts, 
        train_batch_size=args.train_batch_size, 
        train_mismatch_loss=args.train_mismatch_loss, 
        train_mismatch_weight=args.train_mismatch_weight, 
        model_out_file=args.model_out_file, 
        generated_filenames=args.generated_filenames, 
        typical_decoding=args.typical_decoding, 
        dp_optimization=args.dp_optimization, 
        num_sequences_per_prompt=args.num_sequences_per_prompt,
        epsilon=args.epsilon)


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()

    parser.add_argument('--base-model', type=str)
    parser.add_argument('--base-tokenizer', type=str)
    parser.add_argument('--training-file-path', type=str)
    parser.add_argument('--prompts', type=str, nargs='+')
    parser.add_argument('--train-epochs', type=int)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--train-mismatch-loss', action='store_true')
    parser.add_argument('--train-mismatch-weight', type=float)
    parser.add_argument('--model-out-file', type=str)
    parser.add_argument('--generated-filenames', type=str, nargs='+')
    parser.add_argument('--typical-decoding', action='store_true')
    parser.add_argument('--dp-optimization', action='store_true')
    parser.add_argument('--num-sequences-per-prompt', type=int, nargs='+')
    parser.add_argument('--epsilon', type=float)

    args = parser.parse_args()

    run(args)