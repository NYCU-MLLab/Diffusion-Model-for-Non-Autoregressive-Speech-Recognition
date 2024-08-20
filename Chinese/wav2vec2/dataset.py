import random

import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class LibriSpeech(Dataset):

    def __init__(self, path, tokenizer, train):
        self.df = pd.read_csv(path)

        self.features = []

        for feature_file in tqdm(self.df.feature_file, ncols=0):
            self.features.append(torch.load(feature_file))

        self.tokenizer = tokenizer

        if train:
            self.logits = [torch.zeros(feature.size(0), 768) for feature in self.features]
            self.alignments = [torch.zeros(300, feature.size(0), dtype=torch.int16) for feature in self.features]

        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature = self.features[idx]

        text = self.df.text[idx]
        text = self.tokenizer.encode(text)

        if self.train:
            logit = self.logits[idx]

            alignment = self.alignments[idx]
            alignment = alignment[random.randrange(alignment.size(0))].to(torch.int64)

        if self.train:
            return feature, feature.size(0), text, text.size(0), logit, alignment
        else:
            return feature, feature.size(0), text, text.size(0)



def collate_fn(batch, train):
    if train:
        features, feature_lengths, texts, text_lengths, logits, alignments = zip(*batch)
    else:
        features, feature_lengths, texts, text_lengths = zip(*batch)

    features = pad_sequence(features, batch_first=True)
    attention_mask = torch.arange(max(feature_lengths))[None, :] < torch.tensor(feature_lengths)[:, None]
    input_lengths = torch.tensor(feature_lengths)

    texts = torch.cat(texts, dim=0)
    target_lengths = torch.tensor(text_lengths)

    if train:
        logits = pad_sequence(logits, batch_first=True)
        alignments = pad_sequence(alignments, batch_first=True)

    if train:
        return features, attention_mask, input_lengths, texts, target_lengths, logits, alignments
    else:
        return features, attention_mask, input_lengths, texts, target_lengths



def main():
    pass



if __name__ == '__main__':
    main()
