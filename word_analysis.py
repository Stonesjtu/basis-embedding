import torch
import data

###############################################################################
# Load data
###############################################################################

base_name = 'swb'

corpus = data.Corpus(
    path='./data/'+base_name,
    dict_path='./data/'+base_name+'/vocab.txt',
    batch_size=1,
    shuffle=True,
    pin_memory=False,
)

input_coord = torch.load(base_name+'.encoder'+'.coord')
output_coord = torch.load(base_name+'.encoder'+'.coord')

idx2word = corpus.train.dataset.dictionary.idx2word
word2idx = corpus.train.dataset.dictionary.word2idx
coordinates = input_coord
print('Input words to find similarity')
from utils import get_similarity_topk
from itertools import zip_longest
try:
    while True:
        word = input('-->')
        if word in word2idx:
            idx = word2idx[word]
            val, idx = get_similarity_topk(coordinates[idx], coordinates)
            for v, i in zip_longest(val, idx):
                print(v.data[0], '\t', idx2word[i.data[0]])
        else:
            print('word not exists')
except KeyboardInterrupt:
    pass
