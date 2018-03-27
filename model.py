import torch.nn as nn
import torch
from torch.autograd import Variable
from basis_embedding import BasisEmbedding


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

        # block
        self.cm_mask = torch.ones(nhid, nhid).cuda()
        nblock = basis
        block_size = nhid / nblock
        for i in range(nblock):
            n = int(i * block_size)
            m = int((i+1) * block_size)
            self.cm_mask[n:m, n:m].fill_(0)
        self.cm_mask = Variable(self.cm_mask.cuda())
        self.printed = False



    def forward(self, input, target, lengths=None):
        origin_emb = self.encoder(input)
        emb = self.drop(origin_emb)
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        loss = self.criterion(output, target, lengths)
        if not self.encoder.basis and self.training:
            if not self.printed:
                print('haha')
            self.printed = True
            # weight = origin_emb.view(-1, origin_emb.size(2))
            weight = self.encoder.original_matrix
            mean = torch.mean(weight, 1, keepdim=True)
            weight = weight - mean.expand_as(weight)
            cm = torch.mm(weight.t(), weight)
            var = cm.diag().unsqueeze(1)
            var_mat = torch.mm(var, var.t())
            cm_mask = cm * self.cm_mask / var_mat
            return loss + 0.0001 * cm_mask.norm(2)
        return loss
