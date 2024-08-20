import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from transformers import WavLMModel



class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained('facebook/wav2vec2-base')
        self.wavlm.config.mask_time_prob = 0.25

        del self.wavlm.feature_extractor

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 29)
        )

    def ctc(self, hidden_states):
        logits = self.fc(hidden_states)

        return logits

    def forward(self, features, attention_mask):
        hidden_states, extract_features = self.wavlm.feature_projection(features)
        hidden_states = self.wavlm._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=attention_mask
        )

        encoder_outputs = self.wavlm.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        hidden_states = encoder_outputs[0]

        return hidden_states



class PTE(nn.Module):

    def __init__(self, max_len=5000):
        super().__init__()

        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, 768, 2) * (-np.log(10000) / 768))

        pe = torch.empty(max_len, 768, dtype=torch.float32)

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer('pe', pe)

        self.fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 768)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, t):
        x = x + self.pe[:x.size(1)]
        x = x + self.fc(self.pe[10 * t])[:, None, :]

        x = self.dropout(x)

        return x



class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.emb = nn.Embedding(30, 768)

        self.pte = PTE()

        self.dec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 12, dim_feedforward=2 * 768, activation=F.silu, batch_first=True),
            1
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 29)
        )

    def forward(self, hidden_states, attention_mask, x_t, t):
        x_t = self.emb(x_t)
        x_t = x_t + hidden_states

        x_t = self.pte(x_t, t)
        x_t = self.dec(x_t, src_key_padding_mask=~attention_mask)

        logits = self.fc(x_t)

        return logits



class DNAR(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()



def main():
    pass



if __name__ == '__main__':
    main()
