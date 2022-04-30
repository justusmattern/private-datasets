from tqdm import tqdm
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from private_transformers import PrivacyEngine
import argparse

def get_data(path):
    texts = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
            labels.append(int(line.split(' ')[0]))
    
    return texts, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, eos_token):
       self.texts = texts
       self.y = labels
       self.eos_token = eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index] + ' ' + self.eos_token
        label = self.y[index]

        return text, label

def pre_process(texts, labels):
    prompts = []

    for l in labels:
        if l == 1:
            prompts.append("Write a positive review about a good movie:")
        elif l == 0:
            prompts.append("Write a negative review about a bad movie:")
        
    total_texts = [f'{p} {t}' for p, t in zip(prompts, texts)]

    return prompts, total_texts


def run(args):
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.parallelize()
    model.train()
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    optimizer = torch.optim.Adam(model.parameters(), lr = 8e-6)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=1024,
        epochs=args.epochs,
        max_grad_norm=0.1,
        target_epsilon=8,
    )

    privacy_engine.attach(optimizer)

    train_texts, train_labels = get_data('imdb/imdb_train_head.txt')
    train_data = Dataset(train_texts, train_labels, tokenizer.eos_token)

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        print(f'training epoch {epoch}')

        total_loss = 0
        iter = 0
        for texts, labels in tqdm(train_loader):
            iter += 1
            if iter % 100 == 0:
                print(iter)
            prompts, total_texts = pre_process(texts, labels)
            tokenized_prompts = tokenizer(prompts, truncation=True, max_length=1024, return_tensors='pt').input_ids.to('cuda:0')
            tokenized_texts = tokenizer(total_texts, truncation=True, max_length=500, return_tensors='pt', padding=True).input_ids.to('cuda:0')

            lm_loss = model(tokenized_texts, labels=tokenized_texts).loss.unsqueeze(dim=0) # - model(tokenized_prompts, labels=tokenized_text).loss*len()
            #print(logits.shape)
            #print(tokenized_texts.shape)
            #lm_loss = loss_fn(logits.permute(0,2,1), tokenized_texts).mean(dim=1)
            #print('tot texts', total_texts)
            #print(lm_loss)
            #print(lm_loss.shape)
            optimizer.step(loss=lm_loss)
            #optimizer.step()
            total_loss += lm_loss.item()

        print('total lm loss', total_loss/len(train_data))
        model.save_pretrained(f'gpt2_fewshot_epoch{epoch}')


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='gpt2', help='huggingface model name')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='huggingface model name')
    parser.add_argument('--epochs', type=int, default=3, help='number of finetuning epochs')
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()
    run(args)



 

        
