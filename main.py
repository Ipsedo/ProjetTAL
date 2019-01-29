import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import ModelConv
import torch as th
import torch.nn as nn


if __name__ == "__main__":
    sents, ners, ints = load_file.readatis('data-atis/all_atis.iob')

    nb_ints = len(set(ints))

    max_len_sents = 100

    print(max_len_sents)

    sents, ners = prep_data.padd_sents_ners(max_len_sents, sents, ners)

    voc_sents = prep_data.make_vocab(sents)
    voc_ners = prep_data.make_vocab(ners)
    voc_ints = prep_data.make_vocab_ints(ints)

    print(len(voc_sents))
    print(len(voc_ners))

    sents_idx = prep_data.word_to_idx(voc_sents, sents)
    ners_idx = prep_data.word_to_idx(voc_ners, ners)
    ints_idx = prep_data.intents_to_idx(voc_ints, ints)

    sents_idx = prep_data.to_numpy(sents_idx)
    ners_idx = prep_data.to_numpy(ners_idx)
    ints_idx = prep_data.to_numpy(ints_idx)

    print(sents_idx.shape)
    print(ners_idx.shape)

    X = prep_data.to_long_tensor(sents_idx)
    Y = prep_data.to_long_tensor(ints_idx)

    m = ModelConv(len(voc_sents), max_len_sents, nb_ints)
    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adagrad(m.parameters(), lr=1e-1)

    nb_epoch = 10
    batch_size = 32
    nb_batch = int(X.size(0) / batch_size)

    for e in range(nb_epoch):
        sum_loss = 0
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X.size(0) else X.size(0)

            x = X[i_min:i_max]
            y = Y[i_min:i_max]

            optim.zero_grad()

            out = m(x)
            loss = loss_fn(out, y)
            optim.step()

            sum_loss += loss.item()
        print("Epoch %s, loss = %f" % (e, sum_loss / nb_batch))
