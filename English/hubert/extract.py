from pathlib import Path

import pandas as pd

from tqdm import tqdm

import torch
import torchaudio

from transformers import HubertModel

from tokenizer import manual_seed



def make_dataframe(root):
    root = Path(root)

    for split in ['train-clean-100', 'dev-clean', 'dev-other', 'test-clean', 'test-other']:
        sub_root = root / split

        speech_files = sorted(sub_root.rglob('*.flac'))
        text_files = sorted(sub_root.rglob('*.txt'))

        texts = []

        for text_file in text_files:
            with text_file.open() as fp:
                for line in fp.readlines():
                    stem, text = line.strip().split(maxsplit=1)
                    texts.append(text)

                    assert stem == speech_files[len(texts) - 1].stem

        df = pd.DataFrame({'speech_file': speech_files, 'text': texts})
        df.to_csv(f'{root}/{split}.csv', index=False)



@torch.inference_mode()
def extract_feature(root, device='cpu'):
    model = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(device)

    for split in ['train-clean-100', 'dev-clean', 'dev-other', 'test-clean', 'test-other']:
        df = pd.read_csv(f'{root}/{split}.csv')

        feature_files = []

        for speech_file in tqdm(df.speech_file, ncols=0):
            speech, sr = torchaudio.load(speech_file)
            speech = (speech - torch.mean(speech, dim=-1)) / torch.std(speech, dim=-1, correction=0)

            assert sr == 16000

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
