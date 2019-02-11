import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import SuperNN
import torch as th
import torch.nn as nn
import sys

if __name__ == "__main__":

    use_cuda = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "cuda":
            use_cuda = True

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

    nb_train = 3500
    nb_dev = 800

    X_train = X[:nb_train]
    Y_ints_train = Y_ints[:nb_train]
    Y_ners_train = Y_ners[:nb_train]

    X_dev = X[nb_train:nb_train+nb_dev]
    Y_ints_dev = Y_ints[nb_train:nb_train+nb_dev]
    Y_ners_dev = Y_ners[nb_train:nb_train+nb_dev]

    print(X.size())
    print(Y_ints.size())
    print(Y_ners.size())

    nb_epoch = 20
    batch_size = 32
    nb_batch = int(X_train.size(0) / batch_size)

    #m = ModelConv(len(voc_sents), max_len_sents, nb_ints, voc_sents[prep_data.padding_sents])
    m = SuperNN(max_len_sents, len(voc_sents), batch_size, len(voc_ners), len(voc_ints), voc_sents[prep_data.padding_sents])
    loss_fn_ints = nn.CrossEntropyLoss()
    loss_fn_ners = nn.CrossEntropyLoss()

    if use_cuda:
        m.cuda()
        loss_fn_ners.cuda()
        loss_fn_ints.cuda()

    optim = th.optim.Adagrad(m.parameters(), lr=1e-3)

    for e in range(nb_epoch):
        sum_loss_ints = 0
        sum_loss_ners = 0
        m.train()
        for i in range(nb_batch):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X_train.size(0) else X_train.size(0)

            x = X_train[i_min:i_max]
            y_ints = Y_ints_train[i_min:i_max]
            y_ners = Y_ners_train[i_min:i_max].view(-1)

            if use_cuda:
                x, y_ints, y_ners = x.cuda(), y_ints.cuda(), y_ners.cuda()

            optim.zero_grad()

            out_ners, out_ints = m(x)

            loss_ints = loss_fn_ints(out_ints, y_ints)

            index = y_ners != voc_ners[prep_data.padding_ners]
            loss_ners = loss_fn_ners(out_ners.view(-1, len(voc_ners))[index, :], y_ners[index])

            th.sum(loss_ints + loss_ners).backward()
            optim.step()

            sum_loss_ints += loss_ints.cpu().item()
            sum_loss_ners += loss_ners.cpu().item()
        print("Epoch %s, loss_ners = %f, loss_ints = %f" % (e, sum_loss_ners / nb_batch, sum_loss_ints / nb_batch))

        m.eval()
        sum_ints = 0
        sum_ners = 0

        nb_batch_dev = int(nb_dev / batch_size)

        for i in range(nb_batch_dev):
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X_dev.size(0) else X_dev.size(0)

            x = X_dev[i_min:i_max]
            ners = Y_ners_dev[i_min:i_max].view(-1)
            ints = Y_ints_dev[i_min:i_max]

            if use_cuda:
                x, ners, ints = x.cuda(), ners.cuda(), ints.cuda()

            out_ners, out_ints = m(x)

            out_ners = out_ners.view(-1, len(voc_ners)).argmax(dim=1)
            out_ints = out_ints.argmax(dim=1)

            index = ners != voc_ners[prep_data.padding_ners]
            tmp = out_ners[index] == ners[index]
            nb_correct_ners = tmp.sum().cpu().item() / ners[index].size(0)

            nb_correct_ints = (out_ints == ints).sum().cpu().item()

            sum_ints += nb_correct_ints
            sum_ners += nb_correct_ners

        print("Dev results : ners = %f, ints = %f" % (sum_ners / nb_batch_dev, sum_ints / nb_dev))


