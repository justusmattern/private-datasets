from tqdm import tqdm
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from private_transformers import PrivacyEngine
import argparse
from text_dataset import Dataset
from utils import pre_process, get_data_from_txt

def forward_step(correct_texts, wrong_texts, tokenizer, model, mismatch_loss, mismatch_weight):
    tokenized_texts = tokenizer(correct_texts, truncation=True, max_length=500, return_tensors='pt', padding=True).input_ids.to('cuda:0')
    tokenized_texts_wrong = tokenizer(wrong_texts, truncation=True, max_length=500, return_tensors='pt', padding=True).input_ids.to('cuda:0')

    lm_loss = model(tokenized_texts, labels=tokenized_texts).loss.unsqueeze(dim=0)

    if mismatch_loss:
        lm_loss -= mismatch_weight * model(tokenized_texts_wrong, labels=tokenized_texts_wrong).loss.unsqueeze(dim=0)
    
    return lm_loss


def train_lm(args_model, args_tokenizer, args_epochs, args_prompts, args_batch_size, args_mismatch_loss, args_mismatch_weight, args_model_out, return_results, train_data, train_loader, args_dp_optimization, epsilon):
    tokenizer = GPT2Tokenizer.from_pretrained(args_tokenizer)
    model = GPT2LMHeadModel.from_pretrained(args_model, pad_token_id = tokenizer.eos_token_id)
    model.parallelize()
    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = torch.optim.Adam(model.parameters(), lr = 8e-6)

    if args_dp_optimization:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args_batch_size,
            sample_size=1000,
            epochs=args_epochs,
            max_grad_norm=0.1,
            target_epsilon=epsilon,
        )
        privacy_engine.attach(optimizer)

    for epoch in range(args_epochs):
        print(f'training epoch {epoch}')

        total_loss = 0
        for texts, labels in tqdm(train_loader):
            correct_texts, wrong_texts = pre_process(texts, labels, args_prompts)
            lm_loss = forward_step(correct_texts, wrong_texts, tokenizer, model, args_mismatch_loss, args_mismatch_weight)

            if args_dp_optimization:
                optimizer.step(loss=lm_loss)
            else:
                lm_loss = lm_loss.mean()
                lm_loss.backward()
                optimizer.step()

            total_loss += lm_loss.item()

        print('total language modeling loss', total_loss/len(train_data))
        model.save_pretrained(f'{args_model_out}_epoch{epoch}')

    print()
    print('model training done!')
    print()

    if return_results:
        return model


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='gpt2', help='huggingface model name')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='huggingface model name')
    parser.add_argument('--epochs', type=int, default=3, help='number of finetuning epochs')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--mismatch-loss', action='store_true')
    parser.add_argument('--model-out', type=str)
    parser.add_argument('--prompts', type=str, nargs='+', default=["Write a negative review about a bad movie:", "Write a positive review about a good movie:"])
    parser.add_argument('--dp-optimization', action='store_true')
    parser.add_argument('--mismatch-loss-weight', type='float')
    args = parser.parse_args()

    train_texts, train_labels = get_data_from_txt('imdb/imdb_train_head.txt')
    train_data = Dataset(train_texts, train_labels, '<|endoftext|>')
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=args.batch_size)

    train_lm(args_model = args.model, 
            args_tokenizer = args.tokenizer, args_epochs = args.epochs, args_prompts = args.prompts,
            args_batch_size = args.batch_size, args_mismatch_loss = args.mismatch_loss, 
            args_mismatch_weight = args.mismatch_loss_weight, args_model_out = args.model_out, return_results = False,
            train_data=train_data, train_loader=train_loader, args_dp_optimization=args.dp_optimization)



 

