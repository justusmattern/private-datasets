from text_dataset import Dataset
import torch


def get_data_from_txt(path: str):
    texts = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            texts.append(' '.join(line.split(' ')[1:]).replace('\n', ''))
            labels.append(int(line.split(' ')[0]))
    
    return texts, labels


def pre_process(texts, labels, prompts):
    correct_texts = []
    wrong_texts = []

    for text, label in zip(texts, labels):
        correct = prompts[label] + ' ' + text
        correct_texts.append(correct)

        wrong = [prompts[l] + ' ' + text for l in range(len(prompts)) if l != label]
        wrong_texts.extend(wrong)

    return correct_texts, wrong_texts


def prepare_training_data(path, batch_size):

    print()
    print(f'loading training data from {path}...')
    print()

    train_texts, train_labels = get_data_from_txt(path)

    assert(len(train_texts) == len(train_labels))

    train_data = Dataset(train_texts, train_labels, '<|endoftext|>')
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size)
    
    print()
    print(f'training data loaded, {len(train_data)} samples')
    print()
    return train_data, train_loader


def write_to_file(texts, file):
    with open(file, 'w') as f:
        for sample in texts:
            f.write(sample.replace('\n', ' ')+'\n')