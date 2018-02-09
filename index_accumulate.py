import torch
from torch.autograd import Function, Variable

class IndexAccumulate(Function):
    """A function to decode the output of basis linear into normal linear

    Input:
        - partial_score: `(N_b, N_c, N)`
        - aux_codebook: a transformed codebook for flatten partial score

    Return:
        - output: `(V, N)`

    """
    @staticmethod
    def forward(ctx, partial_score, aux_codebook):
        ctx.save_for_backward(aux_codebook)
        ctx.num_basis, ctx.num_cluster, ctx.seq_len = partial_score.size()
        ctx.vocab = aux_codebook.size(0) // ctx.num_basis
        flatten_partial_score = partial_score.view(ctx.num_basis*ctx.num_cluster, ctx.seq_len)
        partial_score_per_word = flatten_partial_score.index_select(0, aux_codebook)
        partial_score = partial_score_per_word.view(ctx.num_basis, ctx.vocab, ctx.seq_len)
        output = partial_score.sum(0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aux_codebook, = ctx.saved_variables
        expanded_grad_output = grad_output.unsqueeze(0).expand(ctx.num_basis, ctx.vocab, ctx.seq_len).contiguous().view(-1, ctx.seq_len)
        grad_input = Variable().type_as(grad_output).resize_(ctx.num_basis*ctx.num_cluster, ctx.seq_len).zero_()
        grad_input.index_add_(0, aux_codebook, expanded_grad_output)
        return grad_input.view(ctx.num_basis, ctx.num_cluster, ctx.seq_len), None


index_accumulate = IndexAccumulate.apply
