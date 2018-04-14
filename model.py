import torch.nn as nn
import torch
from torch.autograd import Variable
from basis_embedding import BasisEmbedding


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder.

    Parameters:
        - ntoken: vocabulary size of input embedding
        - ninp: embedding size
        - nhid: hidden size of LSTM layer
        - nlayers: #layers of LSTM
        - criterion: loss criterion of target word and rnn output
        - dropout: dropout rate of input embedding and lstm layer
        - basis: #input basis
        - num_clusters: #clusters per basis
        - blocked_weight: the weight of blocked loss during pretraining
        - blocked_output: indicates if the output is pre-trained with blocked loss
    """

    def __init__(self,
                 ntoken,
                 ninp,
                 nhid,
                 nlayers,
                 criterion=None,
                 dropout=0.5,
                 basis=0,
                 num_clusters=400,
                 blocked_weight=0,
                 blocked_output=False,
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
        self.blocked_output= blocked_output
        def build_block(dim, basis):
            cm_mask = torch.ones(dim, dim).cuda()
            nblock = basis
            block_size = dim / nblock
            for i in range(nblock):
                n = int(i * block_size)
                m = int((i+1) * block_size)
                cm_mask[n:m, n:m].fill_(0)
            return cm_mask

        if blocked_weight != 0:
            self.cm_mask = Variable(build_block(nhid, basis).cuda())
            self.weight = blocked_weight

    def blocked_loss(self, weight):
        mean = torch.mean(weight, 1, keepdim=True)
        weight = weight - mean.expand_as(weight)
        cm = torch.mm(weight.t(), weight)
        # std = cm.diag().unsqueeze(1).sqrt()
        # var_mat = torch.mm(std, std.t())
        cm_mask = cm * self.cm_mask
        return cm_mask.norm(2)

    def forward(self, input, target, lengths=None):
        origin_emb = self.encoder(input)
        emb = self.drop(origin_emb)
        output, unused_hidden = self.rnn(emb)
        output = self.drop(output)
        loss = self.criterion(output, target, lengths)

        # the blocked loss is used only at pre-trainging
        if not self.encoder.basis and self.training and self.weight != 0:
            # weight = origin_emb.view(-1, origin_emb.size(2))
            weight = self.encoder.original_matrix
            loss += self.weight * self.blocked_loss(weight)
            if self.blocked_output:
                weight = self.criterion.decoder.original_matrix
                loss += self.weight * self.blocked_loss(weight)

        return loss
