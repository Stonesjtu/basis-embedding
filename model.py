import torch.nn as nn
from basis import BasisEmbedding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 ninp,
                 nhid,
                 nlayers,
                 criterion=None,
                 dropout=0.5,
                 tie_weights=False,
                 basis=0,
                 num_clusters=400,
                 ):
        super(RNNModel, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)
        if basis != 0:
            self.encoder = BasisEmbedding(
                ntoken, ninp, basis, num_clusters,
            )
        else:
            self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn = nn.LSTM(
            ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.criterion = criterion


    def forward(self, input, target, lengths=None):
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        loss = self.criterion(output, target, lengths)
        return loss
