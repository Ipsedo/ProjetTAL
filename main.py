import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import SuperNN
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
    Y_ints = prep_data.to_long_tensor(ints_idx)
    Y_ners = prep_data.to_long_tensor(ners_idx)

    print(X.size())
    print(Y_ints.size())
    print(Y_ners.size())

    nb_epoch = 10
    batch_size = 32
    nb_batch = int(X.size(0) / batch_size)

    #m = ModelConv(len(voc_sents), max_len_sents, nb_ints, voc_sents[prep_data.padding_sents])
    m = SuperNN(max_len_sents, len(voc_sents), batch_size, len(voc_ners), len(voc_ints), voc_sents[prep_data.padding_sents])
    loss_fn_ints = nn.CrossEntropyLoss()
    loss_fn_ners = nn.CrossEntropyLoss()
    optim = th.optim.Adagrad(m.parameters(), lr=1e-3)

    for e in range(nb_epoch):
        sum_loss_ints = 0
        sum_loss_ners = 0
        m.train()
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X.size(0) else X.size(0)

            x = X[i_min:i_max]
            y_ints = Y_ints[i_min:i_max]
            y_ners = Y_ners[i_min:i_max]

            optim.zero_grad()

            out_ners, out_ints = m(x)

            loss_ints = loss_fn_ints(out_ints, y_ints)
            loss_ints.backward(retain_graph=True)

            optim.step()

            optim.zero_grad()

            loss_ners = loss_fn_ners(out_ners.view(-1, len(voc_ners)), y_ners.view(-1))
            loss_ners.backward()

            optim.step()

            sum_loss_ints += loss_ints.item()
            sum_loss_ners += loss_ners.item()
        print("Epoch %s, loss_ints = %f, loss_ners = %f" % (e, sum_loss_ints / nb_batch, sum_loss_ners / nb_batch))
