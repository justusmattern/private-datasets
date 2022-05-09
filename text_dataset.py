import torch

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