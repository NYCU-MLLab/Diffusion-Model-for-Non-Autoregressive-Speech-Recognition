import json, random

import numpy as np

import torch



class Tokenizer():

    def __init__(self, path='vocab.json'):
        with open(path) as fp:
            self.vocab = json.load(fp)

    def __len__(self):
        return len(self.vocab['itos'])

    def encode(self, strings):
        return torch.tensor([self.vocab['stoi'][s] for s in strings])

    def decode(self, indices):
        return ''.join([self.vocab['itos'][i] for i in indices])

    def batch_decode(self, batch_indices, attention_mask):
        return [self.decode(indices[mask]) for indices, mask in zip(batch_indices, attention_mask)]

    def decode_ctc(self, indices):
        indices = torch.unique_consecutive(indices)
        indices = indices[indices != 0]

        strings = self.decode(indices)
        strings = ' '.join(strings.split())

        return strings

    def batch_decode_ctc(self, batch_indices, attention_mask):
        return [self.decode_ctc(indices[mask]) for indices, mask in zip(batch_indices, attention_mask)]



def manual_seed(seed=2816):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main():
    pass



if __name__ == '__main__':
    main()
