from pathlib import Path

import pandas as pd

from tqdm import tqdm

import torch
import torchaudio

from transformers import HubertModel

from tokenizer import Tokenizer, manual_seed



def make_dataframe(root):
    dataset = []

    for split in ['validated', 'dev', 'test']:
        df = pd.read_table(f'{root}/{split}.tsv')

        dataset.append([df.path.to_list(), df.sentence.to_list()])

    remove = set(dataset[1][0] + dataset[2][0])

    dataset[0] = [list(z) for z in zip(*[[x, y] for x, y in zip(*dataset[0]) if x not in remove])]

    tokenizer = Tokenizer()

    for i in range(len(dataset)):
        dataset[i][0] = [f'{root}/clips/{path}' for path in dataset[i][0]]
        dataset[i][1] = [''.join([s for s in text if s in tokenizer.vocab['stoi']]) for text in dataset[i][1]]

    for i, split in enumerate(['train', 'dev', 'test']):
        df = pd.DataFrame({'speech_file': dataset[i][0], 'text': dataset[i][1]})
        df.to_csv(f'{root}/{split}.csv', index=False)



@torch.inference_mode()
def extract_feature(root, device='cpu'):
    model = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(device)

    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(f'{root}/{split}.csv')

        feature_files = []

        for speech_file in tqdm(df.speech_file, ncols=0):
            speech, sr = torchaudio.load(speech_file)
            speech = torchaudio.functional.resample(speech, sr, 16000)
            speech = (speech - torch.mean(speech, dim=-1)) / torch.std(speech, dim=-1, correction=0)

            extract_features = model.feature_extractor(speech.to(device))
            extract_features = extract_features.transpose(1, 2)

            path = Path('features') / Path(speech_file).relative_to(root).parent
            path.mkdir(parents=True, exist_ok=True)
            path = path / f'{Path(speech_file).stem}.pt'

            torch.save(extract_features[0].cpu(), path)

            feature_files.append(path)

        df['feature_file'] = feature_files
        df.to_csv(f'{root}/{split}.csv', index=False)



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    manual_seed()

    make_dataframe('datasets')
    extract_feature('datasets', device=device)



if __name__ == '__main__':
    main()
