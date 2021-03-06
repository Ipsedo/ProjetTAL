import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import SuperNN
import torch as th
import torch.nn as nn
import sys
import pickle
from tqdm import tqdm

if __name__ == "__main__":

    # Récupération de l'argument pour lance le modèle avec CUDA
    use_cuda = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "cuda":
            use_cuda = True
    print("Use cuda :", use_cuda)

    # Lecture des données
    sents, ners, ints = load_file.readatis('data-atis/all_atis.iob')

    # Récupération du nombre d'intent différent
    nb_ints = len(set(ints))

    # Taille de phrase maximum = 100
    max_len_sents = 100

    print(max_len_sents)

    # Ajout de padding pour les phrases et ners
    sents, ners = prep_data.padd_sents_ners(max_len_sents, sents, ners)

    # Création des trois vocabulaires (sentences, ners, intents)
    voc_sents = prep_data.make_vocab(sents)
    voc_ners = prep_data.make_vocab(ners)
    voc_ints = prep_data.make_vocab_ints(ints)

    # Sauvegarde des vocabulaires
    vocab_backup = 'vocab.pkl'
    vocab_b_open = open(vocab_backup, 'wb')
    pickle.dump(voc_sents, vocab_b_open)
    pickle.dump(voc_ners, vocab_b_open)
    pickle.dump(voc_ints, vocab_b_open)
    vocab_b_open.close()

    print(len(voc_sents))
    print(len(voc_ners))

    # Passage de mots vers indices
    sents_idx = prep_data.word_to_idx(voc_sents, sents)
    ners_idx = prep_data.word_to_idx(voc_ners, ners)
    ints_idx = prep_data.intents_to_idx(voc_ints, ints)

    # lists vers numpy.ndarray
    sents_idx = prep_data.to_numpy(sents_idx)
    ners_idx = prep_data.to_numpy(ners_idx)
    ints_idx = prep_data.to_numpy(ints_idx)

    print(sents_idx.shape)
    print(ners_idx.shape)

    # numpy.ndarray vers torch.Tensor
    X = prep_data.to_long_tensor(sents_idx)
    Y_ints = prep_data.to_long_tensor(ints_idx)
    Y_ners = prep_data.to_long_tensor(ners_idx)

    # 4478 données d'apprentissage
    # 500 données de test
    nb_train = 4478
    nb_dev = 500

    # Séparation des données d'apprentissage
    X_train = X[:nb_train]
    Y_ints_train = Y_ints[:nb_train]
    Y_ners_train = Y_ners[:nb_train]

    # Séparation des données de test
    X_dev = X[nb_train:nb_train+nb_dev]
    Y_ints_dev = Y_ints[nb_train:nb_train+nb_dev]
    Y_ners_dev = Y_ners[nb_train:nb_train+nb_dev]

    print(X.size())
    print(Y_ints.size())
    print(Y_ners.size())

    # Nombre d'epoch = 15
    nb_epoch = 15

    # Taille de batch égale à 32 phrases
    batch_size = 32
    nb_batch = int(X_train.size(0) / batch_size)

    # Création du modèle
    m = SuperNN(max_len_sents, len(voc_sents), batch_size, len(voc_ners), len(voc_ints), voc_sents[prep_data.padding_sents])

    # Création des deux fonctions de perte
    loss_fn_ints = nn.CrossEntropyLoss()
    loss_fn_ners = nn.CrossEntropyLoss()

    # Passage sur CUDA si nécéssaire
    if use_cuda:
        m.cuda()
        loss_fn_ners.cuda()
        loss_fn_ints.cuda()

    # Optimiseur Adagrad
    optim = th.optim.Adagrad(m.parameters(), lr=1e-3)

    # Boucle principale
    for e in range(nb_epoch):
        sum_loss_ints = 0
        sum_loss_ners = 0

        m.train()

        # Itération sur les batch
        for i in tqdm(range(nb_batch)):
            # Définition des bornes inférieure et superieure du batch
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X_train.size(0) else X_train.size(0)

            # Récupération des données du batch courrant
            x = X_train[i_min:i_max]
            y_ints = Y_ints_train[i_min:i_max]
            y_ners = Y_ners_train[i_min:i_max].view(-1)

            # PAssage sur CUDA si nécéssaire
            if use_cuda:
                x, y_ints, y_ners = x.cuda(), y_ints.cuda(), y_ners.cuda()

            # Remise à zéro du gradient
            optim.zero_grad()

            # Forward
            out_ners, out_ints = m(x)

            # Calcul de la loss pour les intents
            loss_ints = loss_fn_ints(out_ints, y_ints)

            # Calcul de la loss pour les ners
            # On prend soin de retirer les prédictions de mots de padding
            # index contient True à l'indice i si le ieme mot n'est pas du padding (False sinon)
            index = y_ners != voc_ners[prep_data.padding_ners]
            loss_ners = loss_fn_ners(out_ners.view(-1, len(voc_ners))[index, :], y_ners[index])

            # On additionne les deux loss et on procède au backward du modèle (calcul gradient)
            th.sum(loss_ints + loss_ners).backward()

            # Les paramètres sont mis à jour pour l'optimiseur
            optim.step()

            sum_loss_ints += loss_ints.cpu().item()
            sum_loss_ners += loss_ners.cpu().item()
        print("Epoch %s, loss_ners = %f, loss_ints = %f" % (e, sum_loss_ners / nb_batch, sum_loss_ints / nb_batch))

        # Partie evaluation dev set
        m.eval()
        sum_ints = 0
        sum_ners = 0

        nb_batch_dev = int(nb_dev / batch_size)

        # Boucle sur les batch de developpement
        for i in range(nb_batch_dev):
            # Bornes inféreieure et supérieure du batch courrant
            i_min = i * batch_size
            i_max = (i + 1) * batch_size
            i_max = i_max if i_max < X_dev.size(0) else X_dev.size(0)

            # Récupération du batch courrant
            x = X_dev[i_min:i_max]
            ners = Y_ners_dev[i_min:i_max].view(-1)
            ints = Y_ints_dev[i_min:i_max]

            # Passage vers CUDA
            if use_cuda:
                x, ners, ints = x.cuda(), ners.cuda(), ints.cuda()

            # Prédiction du modèle
            out_ners, out_ints = m(x)

            # On prend l'argmax pour connaitre la prédiction du modèle
            out_ners = out_ners.view(-1, len(voc_ners)).argmax(dim=1)
            out_ints = out_ints.argmax(dim=1)

            # Filtrage des mots de padding
            index = ners != voc_ners[prep_data.padding_ners]
            # tmp = nombre de réponse correct
            tmp = out_ners[index] == ners[index]
            # On divise par le nombre de mots n'étants pas du padding
            nb_correct_ners = tmp.sum().cpu().item() / ners[index].size(0)

            # Calcul du nombre de bonnes réponse pour les intents
            nb_correct_ints = (out_ints == ints).sum().cpu().item()

            sum_ints += nb_correct_ints
            sum_ners += nb_correct_ners

        print("Dev results : ners = %f, ints = %f" % (sum_ners / nb_batch_dev, sum_ints / nb_dev))

    # Sauvegarde du modèle
    backup_model = "backup_model.pkl"
    open_backup = open(backup_model, 'wb')
    pickle.dump(m, open_backup)
    open_backup.close()



