import time, json

import pandas as pd

from tqdm import tqdm

import torch
import torchaudio

import jiwer

from transformers import WavLMModel

from tokenizer import Tokenizer, manual_seed
from diffusion import Diffuser
from model import DNAR



@torch.inference_mode()
def test(tokenizer, df, diffuser, feature_extractor, model, device):
    model.eval()
    
    texts_true = []
    texts_pred = []

    total = 0

    for speech_file, text in tqdm(zip(df.speech_file, df.text), total=len(df), ncols=0):
        speech, sr = torchaudio.load(speech_file)
        speech = (speech - torch.mean(speech, dim=-1)) / torch.std(speech, dim=-1, correction=0)

        assert sr == 16000

        extract_features = feature_extractor(speech.to(device))
        extract_features = extract_features.transpose(1, 2)

        attention_mask = torch.full([1, extract_features.size(1)], True, device=device)

        hidden_states = model.encoder(extract_features, attention_mask)

        x_t = torch.full([hidden_states.size(0), hidden_states.size(1)], diffuser.classes - 1, device=device)
        x_t_onehot = diffuser.one_hot(x_t)

        for timestep in range(diffuser.timesteps, 0, -1):
            timesteps = torch.full([x_t.size(0)], timestep, device=device)

            x_t_mask = (x_t == diffuser.classes - 1)

            logits = model.decoder(hidden_states, attention_mask, x_t, timesteps)

            x_0_pred = torch.where(x_t_mask, torch.argmax(logits, dim=-1), x_t)
            x_0_pred_onehot = diffuser.one_hot(x_0_pred)

            x_t_prob = diffuser.reverse(x_t_onehot, x_0_pred_onehot, timesteps)

            x_t = diffuser.sample(x_t_prob)
            x_t_onehot = diffuser.one_hot(x_t)

        texts_true.append(text)
        texts_pred.append(tokenizer.decode_ctc(x_t[0]))

        total += speech.size(1) / sr

    return jiwer.wer(texts_true, texts_pred), total



def main():
    with open('checkpoints/log/model_log.json') as fp:
        log = json.load(fp)

    epoch, wer = min(enumerate(log), key=lambda x: x[1])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer()

    df = pd.read_csv('datasets/test-other.csv')

    diffuser = Diffuser(200, len(tokenizer) + 1, device=device)

    feature_extractor = WavLMModel.from_pretrained('microsoft/wavlm-base').feature_extractor.to(device)

    model = DNAR().to(device)
    model.load_state_dict(torch.load(f'checkpoints/model_{epoch + 1:03d}.pt'))

    manual_seed()

    start = time.time()

    wer, total = test(tokenizer, df, diffuser, feature_extractor, model, device)

    end = time.time()

    print(f'wer: {wer:.2%}')
    print(f'rtf: {(end - start) / total * 1000:.1f}')



if __name__ == '__main__':
    main()
