import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from basis_embedding import BasisEmbedding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 rnn_type,
                 ntoken,
                 ninp,
                 nhid,
                 nlayers,
                 dropout=0.5,
                 tie_weights=False,
                 basis=None,
                 num_clusters=400,
                 load_input_embedding=None,
                 load_output_embedding=None,
                 ):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        if basis:
            self.encoder = BasisEmbedding(
                ntoken, ninp, basis, num_clusters,
                preload_weight_path=load_input_embedding
            )
        else:
            self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, lengths=None):
        emb = self.drop(self.encoder(input))
        emb = self.encoder(input)
        if lengths is not None:
            emb = pack_padded_sequence(emb, list(lengths), batch_first=True)
        output, unused_hidden = self.rnn(emb)
        if isinstance(output, PackedSequence):
            output, _ = pad_packed_sequence(output, batch_first=True)
        # output = self.drop(output)
        return output
