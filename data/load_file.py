import pandas as pd


def readatis(filename='data/atis-2.train.w-intent.iob'):
    data = pd.read_csv(filename, sep='\t', header=None)
    sents = [s.split() for s in data[0].tolist()]
    ners = [s.split() for s in data[1].tolist()]
    # remplacer les chiffres de sents par #
    for i, sent in enumerate(sents):
        sent = ' '.join(sent)
        for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            sent = sent.replace(d, '#')
        sents[i] = sent.split()
    # l'intent est le dernier élément de ners
    ints = [s[-1] for s in ners]
    assert(len(sents) == len(ints))
    return sents, ners, ints

