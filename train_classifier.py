import argparse
import torch
import transformers
import torch.optim as optim
from transformers import BertTokenizer, BertModel

parser = argparse.ArgumentParser()
parser.add_argument('--gen_data', action='store_true')
parser.add_argument('--device', type=int)
parser.add_argument('--pos-file', type=str, default='')
parser.add_argument('--neg-file', type=str, default='')
args = parser.parse_args()

if not args.gen_data:
   print('using real data')
   train_texts = []
   train_labels = []
   with open('imdb/imdb_train_head.txt', 'r') as f:
       for line in f:
           train_texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
           train_labels.append(int(line.split(' ')[0]))

if args.gen_data:
   print('using synth data')
   train_texts = []
   train_labels = []
   with open(args.pos_file, 'r') as f:
     for line in f:
           train_texts.append(' '.join(line.split('good movie:')[1:]).replace('\n', ''))
           train_labels.append(1)
   with open(args.neg_file, 'r') as f:
     for line in f:
           train_texts.append(' '.join(line.split('bad movie:')[1:]).replace('\n', ''))
           train_labels.append(0)

#print('this is gen data')
train_texts = train_texts[:5000]
train_labels = train_labels[:5000]

print(train_texts[:5])
print(train_labels[:5])
test_texts = []
test_labels = []
with open('imdb/imdb_test.txt', 'r') as f:
    for line in f:
        test_texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
        test_labels.append(int(line.split(' ')[0]))

test_texts = test_texts[:5000]
test_labels = test_labels[:5000]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
       self.texts = texts
       self.y = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.y[index]

        return text, label


# HYPERPARAMETERS

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:2")
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 0}


train_data = Dataset(train_texts, train_labels)
test_data = Dataset(test_texts, test_labels)

train_loader = torch.utils.data.DataLoader(train_data, **params)
test_loader = torch.utils.data.DataLoader(test_data, **params)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        #self.bert_model.parallelize()
        self.drop = torch.nn.Dropout(p=0.2)
        self.l1 = torch.nn.Linear(768,2)

    def forward(self, tokenized_text):
        text_rep = self.drop(self.bert_model(tokenized_text).pooler_output)
        out = self.l1(text_rep)

        return out

model = Model()
#model.load_state_dict(torch.load('bert-large-uncased'))
model=model.to(f'cuda:{args.device}')

optimizer = optim.Adam(model.parameters(), lr=1e-5)

loss_f = torch.nn.CrossEntropyLoss()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

for epoch in range(1,20):
    print(f'training epoch {epoch}')

    model.train()
    correct_predictions = 0
    predictions = []
    truth_labels = []
    iter = 0
    for texts, label in train_loader:
        optimizer.zero_grad()
        if iter%10 ==0:
            print(iter)
        iter += 1
        input_tokens = tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=512).input_ids.to(f'cuda:{args.device}')
        #print(input_tokens.shape)
        label = label.long()
        #print('input', input_tokens)
        model_output = model(input_tokens)
        #print('out', model_output.cpu())
        loss = loss_f(model_output.cpu(), label)
        #print('l', loss)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(model_output, dim=1)
        correct_predictions += torch.sum(preds.cpu() == label)
        predictions.extend(preds.tolist())
        truth_labels.extend(label.tolist())


    print('training accuracy ', correct_predictions/len(train_data))
    torch.save(model.state_dict(), f'bert_base_imdb_epoch{epoch}.pt')
    #print(predictions)
    #print(truth_labels)

    model.eval()
    correct_predictions = 0
    predictions = []
    truth_labels = []
    for texts, label in test_loader:
        input_tokens = tokenizer(texts, padding=True, return_tensors='pt', truncation=True, max_length=512).input_ids.to(f'cuda:{args.device}')
        label = label.long()
        model_output = model(input_tokens)


        loss = loss_f(model_output.cpu(), label)

        preds = torch.argmax(model_output, dim=1)
        correct_predictions += torch.sum(preds.cpu() == label)
        predictions.extend(preds.tolist())
        truth_labels.extend(label.tolist())

    print('testing accuracy ', correct_predictions/len(test_data))
