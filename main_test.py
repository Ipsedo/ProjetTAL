import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import SuperNN
import torch as th
import torch.nn as nn
from tqdm import tqdm
import pickle
import sys

if __name__ == "__main__":

    # UTilisation de CUDA ou non
    use_cuda = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "cuda":
            use_cuda = True

    # Chargement des données
    sents, ners, ints = load_file.readatis('data-atis/all_atis.iob')

    max_len_sents = 100

    sents, ners = prep_data.padd_sents_ners(max_len_sents, sents, ners)

    # Chargement des vocabulaires créent pendant la phase d'apprentissage
    vocab = 'vocab.pkl'
    open_vocab = open(vocab, 'rb')
    voc_sents = pickle.load(open_vocab)
    voc_ners = pickle.load(open_vocab)
    voc_ints = pickle.load(open_vocab)

    # Passage vers indices
    sents_idx = prep_data.word_to_idx(voc_sents, sents)
    ners_idx = prep_data.word_to_idx(voc_ners, ners)
    ints_idx = prep_data.intents_to_idx(voc_ints, ints)

    # Passage vers numpy
    sents_idx = prep_data.to_numpy(sents_idx)
    ners_idx = prep_data.to_numpy(ners_idx)
    ints_idx = prep_data.to_numpy(ints_idx)

    # Passage de numpy.ndarray vers torch.Tensor
    X = prep_data.to_long_tensor(sents_idx)
    Y_ints = prep_data.to_long_tensor(ints_idx)
    Y_ners = prep_data.to_long_tensor(ners_idx)

    # Nombre de données de test
    nb_test = X.size(0) - (4478 + 500)

    # Récupération des données de test
    X_test = X[-nb_test:]
    Y_ints_test = Y_ints[-nb_test:]
    Y_ners_test = Y_ners[-nb_test:]

    # Chargement du modèle déjà entrainé
    backup_model = "backup_model.pkl"
    open_backup = open(backup_model, 'rb')
    m = pickle.load(open_backup)

    m.eval()
    sum_ints = 0
    sum_ners = 0
    batch_size = 32

    nb_batch_test = int(nb_test / batch_size)

    # Boucle pour iterer sur les batch de test
    for i in tqdm(range(nb_batch_test)):
        # Bornes du batch
        i_min = i * batch_size
        i_max = (i + 1) * batch_size
        i_max = i_max if i_max < X_test.size(0) else X_test.size(0)

        # Récupération du batch
        x = X_test[i_min:i_max]
        ners = Y_ners_test[i_min:i_max].view(-1)
        ints = Y_ints_test[i_min:i_max]

        # Passage vers CUDA
        if use_cuda:
            x, ners, ints = x.cuda(), ners.cuda(), ints.cuda()

        # Prédiction du modèle
        out_ners, out_ints = m(x)

        # On récupère l'argmax pour obtenir la prédiction
        out_ners = out_ners.view(-1, len(voc_ners)).argmax(dim=1)
        out_ints = out_ints.argmax(dim=1)

        # Filtrage des mots de padding
        index = ners != voc_ners[prep_data.padding_ners]
        # Nombre de bonnes réponses pour les ners
        tmp = out_ners[index] == ners[index]
        nb_correct_ners = tmp.sum().cpu().item() / ners[index].size(0)

        # Nombre de bonnes réponses pour les intents
        nb_correct_ints = (out_ints == ints).sum().cpu().item()

        sum_ints += nb_correct_ints
        sum_ners += nb_correct_ners

    print("Test results : ners = %f, ints = %f" % (sum_ners / nb_batch_test, sum_ints / nb_test))


