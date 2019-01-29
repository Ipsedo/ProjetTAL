import torch as th
import torch.nn as nn


class ModelConv(nn.Module):
    def __init__(self, vocab_size, sent_max_len, nb_class, pad_idx):
        super(ModelConv, self).__init__()

        self.input_size = 100
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, 32, padding_idx=pad_idx)

        self.out_size = int((sent_max_len / 5) / 20)

        self.seq1 = nn.Sequential(nn.Conv1d(32, 64, 5, padding=2),
                                  nn.MaxPool1d(5),
                                  nn.ReLU(),
                                  nn.Conv1d(64, 128, 5, padding=2),
                                  nn.MaxPool1d(20),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(128 * self.out_size, nb_class),
                                  nn.Softmax(dim=1))

    def forward(self, x):
        out = self.emb(x)

        out = out.permute(0, 2, 1)

        out = self.seq1(out).view(-1, self.out_size * 128)
        out = self.seq2(out)

        return out
