import numpy as np
import torch as th

"""
    Mots-clefs pour le padding de phrases et de ners 
"""
padding_sents = "<pad_word>"
padding_ners = "<pad_ners>"


def make_vocab(sents):
    """
    Créer le vocabulaire pour les phrases et les ners
    :param sents: Une liste de liste de string
    :return: Un dictionaire string -> int donnant l'indice associé à un mot
    """
    voc = {}
    for s in sents:
        for w in s:
            if w not in voc:
                voc[w] = len(voc)
    return voc


def make_vocab_ints(intents):
    """
    Créer le vocabulaire pour les intents
    :param intents: Une liste de string
    :return: Un dictionnaire string -> int donnant l'indice associé à un intent
    """
    voc = {}
    for i in intents:
        if i not in voc:
            voc[i] = len(voc)
    return voc


def get_max_len(sents):
    """
    Récupère la longueur maximale d'une liste de phrase
    :param sents: Une liste de liste représentant toutes les phrases
    :return: La longueur maximale des phrases données en arguments
    """
    length = 0
    for s in sents:
        length = len(s) if len(s) > length else length
    return length


def padd_sents_ners(max_len, sents, ners):
    """
    Ajoute du padding pour les phrases et les ners selon une longueur maximale
    :param max_len: La longueur maximale que l'on cherche à obtenir
    :param sents: Les phrases (liste de liste de string)
    :param ners: Les ners (liste de liste de string)
    :return: Un tuple (sent, ners) contenant les phrases et les ners auxquels du padding
        a été rajouté pour une longueur uniforme
    """
    sents_res = []
    ners_res = []
    for s, n in zip(sents, ners):
        s += [padding_sents for _ in range(max_len - len(s))]
        n += [padding_ners for _ in range(max_len - len(n))]
        sents_res.append(s)
        ners_res.append(n)
    return sents_res, ners_res


def word_to_idx(vocab, sents):
    """
    Passe la représentation de mots en chaine de caractère vers un entier représentant son indice
    :param vocab: Le vocabulaire (dictionnaire string -> int) donnant l'indice associé à un mot
    :param sents: La liste de listes de mots à transformer
    :return: La représentation des mots sous leur indice (sous le format de liste de liste d'entier).
    """
    res = []
    for s in sents:
        idxs = []
        for w in s:
            idxs.append(vocab[w])
        res.append(idxs)
    return res


def intents_to_idx(vocab, intents):
    """
    Passe la représentation d'intents en chaine de caractère vers un entier représentant son indice
    :param vocab: Le vocabulaire pour les intents (dictionnaire string -> int)
    :param intents: La liste d'intents (liste de string)
    :return: La représentation des intents sous forme d'indice (liste d'entier)
    """
    res = []
    for i in intents:
        res.append(vocab[i])
    return res


def to_numpy(idxs):
    """
    Pass to numpy
    :param idxs: list [of list] of int
    :return: A numpy ndarray representing the given list
    """
    return np.asarray(idxs)


def to_long_tensor(idxs_np):
    """
    Pass to torch.LongTensor
    :param idxs_np: A numpy ndarray
    :return: A torch.Tensor of type th.long
    """
    return th.Tensor(idxs_np).type(th.long)
