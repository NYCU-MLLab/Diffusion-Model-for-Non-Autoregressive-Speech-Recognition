from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence



def sample_alignment_ones(logits, texts, n_sample=1):
    alignment_list = []

    for logit, text in zip(logits, texts):
        column = torch.empty(2 * text.size(0) + 1, dtype=torch.int64)

        column[0::2] = 0
        column[1::2] = text

        alpha = torch.full([logit.size(0), column.size(0) + 1], -torch.inf)

        alpha[0][0] = logit[0][column[0]]
        alpha[0][1] = logit[0][column[1]]

        index = torch.arange(column.size(0))

        index = torch.where(
            torch.stack([
                    index >= 0,
                    index > 0,
                    torch.cat([index[:2] > 1, column[index[2:]] != column[index[2:] - 2]], dim=0)
                ],
                dim=1
            ),
            torch.stack([index, index - 1, index - 2], dim=1),
            index.size(0)
        )

        for i in range(1, alpha.size(0)):
            alpha[i][:-1] = torch.logsumexp(alpha[i - 1][index.view(-1)].view(-1, 3), dim=1) + logit[i][column]

        alignments = torch.empty(alpha.size(0), n_sample, dtype=torch.int64)

        alignments[-1] = -1 - torch.distributions.categorical.Categorical(
            logits=alpha[-1][[-2, -3]]
        ).sample([n_sample])

        for i in range(-2, -1 - alignments.size(0), -1):
            alignments[i] = alignments[i + 1] - torch.distributions.categorical.Categorical(
                logits=alpha[i][index[alignments[i + 1]].view(-1)].view(-1, 3)
            ).sample()

        alignments = column[alignments.view(-1)].view(-1, n_sample)
        alignments = alignments.transpose(0, 1)

        alignment_list.append(alignments[0])

    return pad_sequence(alignment_list, batch_first=True)



@torch.inference_mode()
def sample_alignment(tokenizer, df, encoder, device, n_sample=300):
    encoder.eval()

    logit_list = []
    alignment_list = []

    for feature_file, text in tqdm(zip(df.feature_file, df.text), total=len(df), ncols=0):
        features = torch.load(feature_file)[None, :, :]
        attention_mask = torch.full([features.size(0), features.size(1)], True)

        hidden_states = encoder(features.to(device), attention_mask.to(device))

        logits = encoder.ctc(hidden_states)
        logits = F.log_softmax(logits, dim=-1)

        logit = logits[0].cpu()

        column = torch.empty(2 * len(text) + 1, dtype=torch.int64)

        column[0::2] = 0
        column[1::2] = tokenizer.encode(text)

        alpha = torch.full([logit.size(0), column.size(0) + 1], -torch.inf)

        alpha[0][0] = logit[0][column[0]]
        alpha[0][1] = logit[0][column[1]]

        index = torch.arange(column.size(0))

        index = torch.where(
            torch.stack([
                    index >= 0,
                    index > 0,
                    torch.cat([index[:2] > 1, column[index[2:]] != column[index[2:] - 2]], dim=0)
                ],
                dim=1
            ),
            torch.stack([index, index - 1, index - 2], dim=1),
            index.size(0)
        )

        for i in range(1, alpha.size(0)):
            alpha[i][:-1] = torch.logsumexp(alpha[i - 1][index.view(-1)].view(-1, 3), dim=1) + logit[i][column]

        alignments = torch.empty(alpha.size(0), n_sample, dtype=torch.int64)

        alignments[-1] = -1 - torch.distributions.categorical.Categorical(
            logits=alpha[-1][[-2, -3]]
        ).sample([n_sample])

        for i in range(-2, -1 - alignments.size(0), -1):
            alignments[i] = alignments[i + 1] - torch.distributions.categorical.Categorical(
                logits=alpha[i][index[alignments[i + 1]].view(-1)].view(-1, 3)
            ).sample()

        alignments = column[alignments.view(-1)].view(-1, n_sample)
        alignments = alignments.transpose(0, 1)

        for alignment in alignments:
            assert tokenizer.decode_ctc(alignment) == text

        logit_list.append(hidden_states[0].cpu())
        alignment_list.append(alignments.to(torch.int16))

    torch.cuda.empty_cache()

    return logit_list, alignment_list
