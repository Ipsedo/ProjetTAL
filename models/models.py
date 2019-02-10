import torch as th
import torch.nn as nn


class ModelConv_old(nn.Module):
    def __init__(self, vocab_size, sent_max_len, nb_class, pad_idx):
        super(ModelConv_old, self).__init__()

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
    def __init__(self, vocab_size, padding_idx, emb_size=16):
        super(Embedding, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)

    def forward(self, x):
        return self.emb(x)


class DoubleLSTM(nn.Module):
    def __init__(self, seq_length, batch_size, out_class, emb_size=16, hidden_size=24):
        super(DoubleLSTM, self).__init__()

        self.seq_length = seq_length
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_class = out_class

        self.lstm_1 = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        #self.lstm_2 = nn.LSTM(self.hidden_size * 2, self.hidden_size * 2 + 8, bidirectional=True, batch_first=True)

        self.fst_h_1 = th.randn(4, batch_size, self.hidden_size)
        self.fst_c_1 = th.randn(4, batch_size, self.hidden_size)

        #self.fst_h_2 = th.randn(2, batch_size, self.hidden_size * 2 + 8)
        #self.fst_c_2 = th.randn(2, batch_size, self.hidden_size * 2 + 8)

        #self.lin = nn.Linear((self.hidden_size * 2 + 8) * 2, self.out_class)
        self.dense = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 6),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size * 6, out_class),
                                   nn.ReLU())

    def forward(self, x):
        s = x.size(0)
        o1, _ = self.lstm_1(x, (self.fst_h_1[:, :s, :], self.fst_c_1[:, :s, :]))
        #o2, _ = self.lstm_2(o1, (self.fst_h_2[:, :s, :], self.fst_c_2[:, :s, :]))
        o3 = self.dense(o1)
        return o3


class ConvModel(nn.Module):
    def __init__(self, out_channels, emb_size=16):
        super(ConvModel, self).__init__()
        self.emb_size = emb_size
        self.out_channels = out_channels
        self.seq = nn.Sequential(nn.Conv1d(self.emb_size, 24, kernel_size=3),
                                 nn.MaxPool1d(3, 2),
                                 nn.ReLU(),
                                 nn.Conv1d(24, out_channels, kernel_size=5),
                                 nn.MaxPool1d(5, 2),
                                 nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.seq(x)
        return out


class SuperNN(nn.Module):
    def __init__(self, seq_length, vocab_size, batch_size, word_out_class, seq_out_class, padding_idx):
        super(SuperNN, self).__init__()

        self.size_seq_pred = (seq_length + 20) * word_out_class

        self.emb = Embedding(vocab_size, padding_idx)
        self.lstm = DoubleLSTM(seq_length, batch_size, word_out_class)
        self.conv = ConvModel(word_out_class)

        #self.drop = nn.Dropout(0.7)

        #self.act_word_pred = nn.Softmax(dim=2)

        self.seq_pred = nn.Linear(self.size_seq_pred, seq_out_class)

    def forward(self, x):
        embedded = self.emb(x)
        out_lstm = self.lstm(embedded)
        out_conv = self.conv(embedded).permute(0, 2, 1)

        out_concat = th.cat((out_lstm, out_conv), 1).view(-1, self.size_seq_pred)
        #out_concat = self.drop(out_concat)

        #out_word_pred = self.act_word_pred(out_lstm)
        out_seq_pred = self.seq_pred(out_concat)

        return out_lstm, out_seq_pred
