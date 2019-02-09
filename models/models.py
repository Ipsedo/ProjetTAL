import torch as th
import torch.nn as nn


class ModelConv_old(nn.Module):
    def __init__(self, vocab_size, sent_max_len, nb_class, pad_idx):
        super(ModelConv, self).__init__()

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


class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_size=16):
        super(Embedding, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, x):
        return self.emb(x)


class DoubleLSTM(nn.Module):
    def __init__(self, seq_length, batch_size, out_class, emb_size=16, hidden_size=16):
        super(DoubleLSTM, self).__init__()

        self.seq_length = seq_length
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_class = out_class

        self.lstm_1 = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.lstm_2 = nn.LSTM(self.hidden_size * 2, self.hidden_size * 2, bidirectional=True, batch_first=True)

        self.fst_h_1 = th.randn(2, batch_size, self.hidden_size)
        self.fst_c_1 = th.randn(2, batch_size, self.hidden_size)

        self.fst_h_2 = th.randn(2, batch_size, self.hidden_size * 2)
        self.fst_c_2 = th.randn(2, batch_size, self.hidden_size * 2)

        self.lin = nn.Linear(self.hidden_size * 4, self.out_class)

    def forward(self, x):
        o1, _ = self.lstm_1(x, (self.fst_h_1, self.fst_c_1))
        o2, _ = self.lstm_2(o1, (self.fst_h_2, self.fst_c_2))
        o3 = self.lin(o2)
        return o3


class ConvModel(nn.Module):
    def __init__(self, seq_length, out_channels, emb_size=16):
        super(ConvModel, self).__init__()
        self.emb_size = emb_size
        self.out_channels = out_channels
        self.seq_length = seq_length
        self.seq = nn.Sequential(nn.Conv1d(self.emb_size, self.out_channels, kernel_size=3),
                                 nn.MaxPool1d(seq_length - 2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.seq(x)
        return out


class SuperNN(nn.Module):
    def __init__(self, seq_length, vocab_size, batch_size, word_out_class, seq_out_class):
        super(SuperNN, self).__init__()

        self.size_seq_pred = (seq_length + 1) * word_out_class

        self.emb = Embedding(vocab_size)
        self.lstm = DoubleLSTM(seq_length, batch_size, word_out_class)
        self.conv = ConvModel(seq_length, word_out_class)

        self.act_word_pred = nn.Softmax(dim=2)

        self.seq_pred = nn.Sequential(nn.Linear(self.size_seq_pred, seq_out_class),
                                      nn.Softmax(dim=1))

    def forward(self, x):
        embedded = self.emb(x)
        out_lstm = self.lstm(embedded)
        out_conv = self.conv(embedded).permute(0, 2, 1)

        out_concat = th.cat((out_lstm, out_conv), 1).view(-1, self.size_seq_pred)

        out_word_pred = self.act_word_pred(out_lstm)
        out_seq_pred = self.seq_pred(out_concat)

        return  out_word_pred, out_seq_pred
