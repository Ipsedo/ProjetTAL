import data.load_file as load_file
import data.prepare_data as prep_data
from models.models import SuperNN
import torch as th
import torch.nn as nn
from tqdm import tqdm
import pickle
import sys

use_cuda = False
if len(sys.argv) > 1:
    if sys.argv[1] == "cuda":
        use_cuda = True

sents, ners, ints = load_file.readatis('data-atis/all_atis.iob')

max_len_sents = 100

sents, ners = prep_data.padd_sents_ners(max_len_sents, sents, ners)

voc_sents = prep_data.make_vocab(sents)
voc_ners = prep_data.make_vocab(ners)
voc_ints = prep_data.make_vocab_ints(ints)

sents_idx = prep_data.word_to_idx(voc_sents, sents)
ners_idx = prep_data.word_to_idx(voc_ners, ners)
ints_idx = prep_data.intents_to_idx(voc_ints, ints)

sents_idx = prep_data.to_numpy(sents_idx)
ners_idx = prep_data.to_numpy(ners_idx)
ints_idx = prep_data.to_numpy(ints_idx)

X = prep_data.to_long_tensor(sents_idx)
Y_ints = prep_data.to_long_tensor(ints_idx)
Y_ners = prep_data.to_long_tensor(ners_idx)

print(X.size(0))

nb_test = 3500+1371

print(nb_test)

X_test = X[nb_test:]
Y_ints_test = Y_ints[nb_test:]
Y_ners_test = Y_ners[nb_test:]

print(X_test.size())
print(Y_ints_test.size())
print(Y_ners_test.size())

#Get the trained model

backup_model = "backup_model.pkl"
open_backup = open(backup_model, 'rb')
m = pickle.load(open_backup)
print(m)

m.eval()
sum_ints = 0
sum_ners = 0
batch_size = 32

nb_batch_test = int(1000 / batch_size)

for i in tqdm(range(nb_batch_test)):
    i_min = i * batch_size
    i_max = (i + 1) * batch_size
    i_max = i_max if i_max < X_test.size(0) else X_test.size(0)

    x = X_test[i_min:i_max]
    ners = Y_ners_test[i_min:i_max].view(-1)
    ints = Y_ints_test[i_min:i_max]

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

print("Test results : ners = %f, ints = %f" % (sum_ners / nb_batch_test, sum_ints / nb_test))


