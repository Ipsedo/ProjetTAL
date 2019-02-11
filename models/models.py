import torch as th
import torch.nn as nn

"""
    Modèle non-utilisé
"""
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


embedding_size = 512
hidden_lstm_size = 256


class Embedding(nn.Module):
    def __init__(self, vocab_size, padding_idx, emb_size=embedding_size):
        """
        Module pour la couche d'embedding
        :param vocab_size: Taille du vocabulaire
        :param padding_idx: Indice du padding (pour remplacer par des vecteurs (de taille self.emb_size) nuls)
        :param emb_size: Taille du vecteur d'embedding
        """
        super(Embedding, self).__init__()

        # Définitions attributs
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        # Définition couche embedding
        self.emb = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=padding_idx)

    def forward(self, x):
        """

        :param x: Un torch.Tensor (torch.LongTensor) de shape = (batch, seq) où les valeurs de seq sont inférieure à
            self.vocab_size
        :return: Un torch.Tensor (torch.FloatTensor) de shape = (batch, seq, self.emb_size) représentant
            l'embedding des mots
        """
        return self.emb(x)


class DoubleLSTM(nn.Module):
    def __init__(self, seq_length, batch_size, out_class, emb_size=embedding_size, hidden_size=hidden_lstm_size):
        """
        Module pour le double stacked LSTM et les deux couches denses
        Utilisé pour les deux taches de classification
        :param seq_length: La taille en nombre de mots d'une phrase
        :param batch_size: La taille de batch (utile pour les premiers vecteurs cachés du LSTM)
        :param out_class: Le nombre de classe pour la classification de mot
        :param emb_size: La taille du vecteur d'embedding
        :param hidden_size: La taille pour les deux vecteurs cachés du LSTM
        """
        super(DoubleLSTM, self).__init__()

        # Définition attributs
        self.seq_length = seq_length
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_class = out_class

        # Création du LSTM :
        # - bi-directionnel
        # - deux couches superposées
        # - l'indice du batch en première position
        # - dropout de probabilité 0.3
        self.lstm_1 = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.3)

        # Création des premiers vecteurs cachés et des premier états du LSTM
        # 4 au total :
        # - 2 pour les deux directons du premier layer du LSTM
        # - encore 2 pour le deuxième layer
        self.fst_h_1 = th.randn(4, batch_size, self.hidden_size)
        self.fst_c_1 = th.randn(4, batch_size, self.hidden_size)

        # Dense layers composés de deux couches linéaires
        # Sortie : nombre de classes de mots
        self.dense = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size * 6),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size * 6, out_class),
                                   nn.ReLU())

    def forward(self, x):
        """
        Fonction de forward du modèle
        :param x: Un torch.Tensor (torch.FloatTensor) des embedding des mots.
            Tensor de shape = (batch, seq, emb) où la taille de batch, la longueur de la séquence
            et la taille d'embedding (emb) doivent être similaires aux informations fournies
            dans le constructeur (sauf pour la taille de batch qui peut être inférieure).
        :return: Un torch.Tensor (torch.FloatTensor) donnant la prédiction par mot
            Tensor de shape (batch, seq, out_class_word)
        """
        # On récupère la taille de batch
        batch_size = x.size(0)

        # On adapte la taille des vecteurs cachés pour prévenir des cas
        # où la taille de batch est inférieure à celle fournie dans le constructeur
        h = self.fst_h_1[:, :batch_size, :]
        c = self.fst_c_1[:, :batch_size, :]

        # On passe ces deux vecteurs vers le gpu si nécéssaire
        if x.is_cuda:
            h, c = h.cuda(), c.cuda()

        # Application de la couche LSTM
        o1, _ = self.lstm_1(x, (h, c))

        # Application de la couche dense
        o2 = self.dense(o1)

        return o2


class ConvModel(nn.Module):
    def __init__(self, out_channels, emb_size=embedding_size):
        """
        Modèle Convolutionnel
        Utilisé pour la classification d'intents à l'aide de DoubleLSTM
        :param out_channels: Nombre de channels de sortie. Dans notre cas pour une raison de compatibilité
            entre ce modèle et le DoubleLSTM, out_channels sera égal aux nombre de classes de mots.
        :param emb_size: La taille du vecteur d'embedding
        """
        super(ConvModel, self).__init__()

        # Définition attributs
        self.emb_size = emb_size
        self.out_channels = out_channels

        # Définitions de la séquence de convolutions et de max-pools
        # Activations ReLU
        self.seq = nn.Sequential(nn.Conv1d(self.emb_size, 24, kernel_size=3),
                                 nn.MaxPool1d(3, 2),
                                 nn.ReLU(),
                                 nn.Conv1d(24, out_channels, kernel_size=5),
                                 nn.MaxPool1d(5, 2),
                                 nn.ReLU())

    def forward(self, x):
        """
        Fonction de forward du modèle convolutionnel
        :param x: Un torch.Tensor (torch.FloatTensor) des embeddings des mots.
            Tensor de shape = (batch, seq, emb).
        :return: Un torch.Tensor (torch.FloatTensor) de shape (batch, out_channels, batch).
        """
        # On a besoin de permuter la dimension des embedding (channels, indice = 2) et celle représentant
        # la séquence de mot (indice = 1) pour coller avec ce que demande PyTorch comme dimension
        # pour la convolution
        x = x.permute(0, 2, 1)

        # On applique les deux couches Convolution - MaxPool - ReLU
        out = self.seq(x)

        return out


class SuperNN(nn.Module):
    def __init__(self, seq_length, vocab_size, batch_size, word_out_class, seq_out_class, padding_idx):
        """
        Combinaison des 3 modèle (Embedding + DoubleLSTM + ConvModel) pour la double prédiction
        :param seq_length: La taille d'une phrase
        :param vocab_size: La taille du vocabulaire
        :param batch_size: La taille d'un batch de phrases
        :param word_out_class: Le nombre de classes de mot
        :param seq_out_class: Le nombre de classes de phrase
        :param padding_idx: L'indice du mot de padding
        """
        super(SuperNN, self).__init__()

        # On définit le nombre de features pour la prédiction de phrase.
        # Concaténation du résultat du LSTM et du modèle à convolution + ravel.
        # nombre de mots = seq_length (DoubleLSTM) + 20 (ConvModel)
        # nombre de features = nombre_de_mots * word_out_class
        self.size_seq_pred = (seq_length + 20) * word_out_class

        # Définition des 3 sous-modèles
        self.emb = Embedding(vocab_size, padding_idx)
        self.lstm = DoubleLSTM(seq_length, batch_size, word_out_class)
        self.conv = ConvModel(word_out_class)

        # Définiton de la couche dense pour la prédiction de phrase
        # Prends en entrée la concatenation de la sortie de LSTM et du modèle à convolution
        self.seq_pred = nn.Linear(self.size_seq_pred, seq_out_class)

    def forward(self, x):
        """
        Forward du modèle principal
        :param x: Un torch.Tensor (torch.LongTensor) de shape = (batch, seq) où les valeurs de seq sont inférieure à
            self.vocab_size
        :return: Un tuple (prediction_par_mot, prediction_phrase)
            Preidtcion_par_mot = torch.Tensor (torch.FloatTensor) contenant la prédiction par mots des phrases données
            en entrée et qui a pour shape (batch, seq, word_out_class).
            Prediction_phrase = torch.Tensor (torch.FloatTensor) contenant la prédiction par phrases de l'ensemble du
            du batch et qui a pout shape (batch, seq_out_class)
        """
        # On calcule les embedding de mots
        embedded = self.emb(x)

        # On récupère la sortie du LSTM
        out_lstm = self.lstm(embedded)

        # Ainsi que celle du modèle à convolutions
        # On remet l'ordre des dimension pour coller à (batch, seq, word_out_class)
        out_conv = self.conv(embedded).permute(0, 2, 1)

        # Les deux résultats (LSTM + Conv) sont concaténés et mis à plat (ravel)
        # -1 pour la taille de batch
        out_concat = th.cat((out_lstm, out_conv), 1).view(-1, self.size_seq_pred)

        # On applique la couche linéaire pour la predictions de phrases
        out_seq_pred = self.seq_pred(out_concat)

        # Le résultats de la prédiction de mot et de phrase sont renvoyés
        return out_lstm, out_seq_pred
