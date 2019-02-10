import numpy as np
import torch as th


padding_sents = "<pad_word>"
padding_ners = "<pad_ners>"


def make_vocab(sents):
    voc = {}
    for s in sents:
        for w in s:
            if w not in voc:
                voc[w] = len(voc)
    return voc


def make_vocab_ints(intents):
    voc = {}
    for i in intents:
        if i not in voc:
            voc[i] = len(voc)
    return voc


def get_max_len(sents):
    length = 0
    for s in sents:
        length = len(s) if len(s) > length else length
    return length


def padd_sents_ners(max_len, sents, ners):
    sents_res = []
    ners_res = []
    for s, n in zip(sents, ners):
        s += [padding_sents for _ in range(max_len - len(s))]
        n += [padding_ners for _ in range(max_len - len(n))]
        sents_res.append(s)
        ners_res.append(n)
    return sents_res, ners_res


def word_to_idx(vocab, sents):
    res = []
    for s in sents:
        idxs = []
        for w in s:
            idxs.append(vocab[w])
        res.append(idxs)
    return res


def intents_to_idx(vocab, intents):
    res = []
    for i in intents:
        res.append(vocab[i])
    return res


def to_numpy(idxs):
    return np.asarray(idxs)


def to_long_tensor(idxs_np):
    return th.Tensor(idxs_np).type(th.long)
