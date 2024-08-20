import torch
from torch.nn import functional as F



class Diffuser():

    LOG_MIN = 1e-30

    def __init__(self, timesteps, classes, dtype=torch.float32, device='cpu'):
        self.timesteps = timesteps
        self.classes = classes

        self.dtype = dtype
        self.device = device

        beta = 1 / (timesteps - torch.arange(1, timesteps + 1, dtype=torch.float64) + 1)
        alpha = 1 - beta

        bar_alpha = torch.cumprod(alpha, dim=0)
        bar_beta = 1 - bar_alpha

        self.beta = beta.to(dtype).to(device)[:, None, None]
        self.alpha = alpha.to(dtype).to(device)[:, None, None]

        self.bar_alpha = bar_alpha.to(dtype).to(device)[:, None, None]
        self.bar_beta = bar_beta.to(dtype).to(device)[:, None, None]

    @staticmethod
    def sample(probs):
        return torch.distributions.categorical.Categorical(probs).sample()

    @staticmethod
    def split(x, l):
        return [xx[:ll] for xx, ll in zip(x, l)]

    @classmethod
    def log(cls, x):
        return torch.log(torch.clamp(x, min=cls.LOG_MIN))

    def one_hot(self, x):
        return F.one_hot(x, num_classes=self.classes).to(self.dtype)

    def forward_from_x_0(self, x_0, t):
        probs_1 = self.bar_alpha[t - 1] * x_0
        probs_2 = self.bar_beta[t - 1] * self.one_hot(torch.tensor(self.classes - 1, device=self.device))

        return probs_1 + probs_2

    def forward_posterior(self, x_t, x_0, t):
        probs_1 = self.alpha[t - 1] * x_t + torch.where(x_t[:, :, -1:] == 1, self.beta[t - 1], 0)
        probs_2 = self.forward_from_x_0(x_0, t - 1)

        return F.normalize(probs_1 * probs_2, p=1, dim=-1)

    def reverse(self, x_t, x_0_pred, t):
        probs = torch.empty_like(x_t)

        probs[t != 1] = self.forward_posterior(x_t[t != 1], x_0_pred[t != 1], t[t != 1])
        probs[t == 1] = x_0_pred[t == 1]

        return probs

    def dfba(self, x_t, x_0_pred, t, x_t_mask, input_lengths, texts, target_lengths, logits_ctc):
        x_t = self.split(x_t, input_lengths)
        x_0_pred = self.split(x_0_pred, input_lengths)
        x_t_mask = self.split(x_t_mask, input_lengths)

        texts = torch.split(texts, target_lengths.tolist(), dim=0)

        loss = []

        for x_t, x_0_pred, t, x_t_mask, text, logit_ctc in zip(x_t, x_0_pred, t, x_t_mask, texts, logits_ctc):
            column = torch.empty(2 * text.size(0) + 1, dtype=torch.int64)

            column[0::2] = 0
            column[1::2] = text

            unique, inverse = torch.unique(column, return_inverse=True)

            probs_true = self.reverse(
                torch.repeat_interleave(x_t[None, :, :], unique.size(0), dim=-2),
                self.one_hot(torch.tile(unique.to(self.device), [1, x_t.size(0)])),
                t[None]
            )[0]
            probs_pred = self.reverse(
                x_t[None, :, :],
                x_0_pred[None, :, :],
                t[None]
            )[0]

            probs_true = torch.reshape(probs_true, [-1, unique.size(0), probs_true.size(-1)])
            probs_pred = probs_pred[:, None, :]

            if t != 1:
                loss_vb = torch.sum(probs_true * (self.log(probs_true) - self.log(probs_pred)), dim=-1)
            else:
                loss_vb = - torch.sum(probs_true * self.log(probs_pred), dim=-1)

            loss_vb = torch.where(x_t_mask[:, None], loss_vb, 0.0)
            loss_vb = loss_vb.cpu()

            logit_ctc = torch.where(x_t_mask[:, None].cpu(), logit_ctc, torch.log(x_t[:, :-1]).cpu())

            alpha = torch.full([logit_ctc.size(0), column.size(0) + 1], -torch.inf)

            alpha[0][0] = logit_ctc[0][column[0]]
            alpha[0][1] = logit_ctc[0][column[1]]

            beta = torch.full([logit_ctc.size(0), column.size(0) + 1], 0.0)

            beta[0][0] = loss_vb[0][inverse[0]] * alpha[0][0].exp()
            beta[0][1] = loss_vb[0][inverse[1]] * alpha[0][1].exp()

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

            for i in range(1, beta.size(0)):
                alpha[i][:-1] = torch.logsumexp(
                    alpha[i - 1][index.view(-1)].view(-1, 3),
                    dim=1
                ) + logit_ctc[i][column]

                beta[i][:-1] = torch.sum(
                    beta[i - 1][index.view(-1)].view(-1, 3),
                    dim=1
                ) * logit_ctc[i][column].exp() + loss_vb[i][inverse] * alpha[i][:-1].exp()

            loss.append((beta[-1][-2] + beta[-1][-3]) / (alpha[-1][-2].exp() + alpha[-1][-3].exp()))

        return (sum(loss) / len(loss)).to(self.device)



def main():
    pass



if __name__ == '__main__':
    main()
