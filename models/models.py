import torch as th
import torch.nn as nn


class ModelConv(nn.Module):
    def __init__(self, vocab_size, sent_max_len, nb_class):
        super(ModelConv, self).__init__()

        self.input_size = 100
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, 32)

        self.out_size = int((sent_max_len / 5) / 20)

        self.seq1 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1),
                                  nn.MaxPool1d(5),
                                  nn.ReLU(),
                                  nn.Conv1d(64, 128, 3, padding=1),
                                  nn.MaxPool1d(20),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(128 * self.out_size, nb_class),
                                  nn.Softmax(dim=1))

    def forward(self, input):
        out = self.emb(input)

        out = out.permute(0, 2, 1)

        out = self.seq1(out).view(-1, self.out_size * 128)
        out = self.seq2(out)

        return out
