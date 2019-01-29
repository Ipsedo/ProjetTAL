import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import ModelConv
import torch as th
import torch.nn as nn


if __name__ == "__main__":
    sents, ners, ints = load_file.readatis('data-atis/all_atis.iob')

    nb_ints = len(set(ints))

    """max_len_sents = prep_data.get_max_len(sents)
    max_len_ners = prep_data.get_max_len(ners)
    assert(max_len_sents == max_len_ners)"""
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
    optim = th.optim.SGD(m.parameters(), lr=1e-3)

    out = m(X[0:10])
    loss = loss_fn(out, Y[0:10])
    loss.backward()
    optim.step()
