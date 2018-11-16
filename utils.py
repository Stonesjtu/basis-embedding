"""Some utilities functions to help abstract the codes"""
import logging

from torch.autograd import Variable

import configargparse


def get_similarity_count(source, target):
    """Get the similarity counts between source vector and target vectors in matrix"""
    similarity_mat = source == target
    similarity_count = similarity_mat.sum(dim=1)
    return similarity_count

def get_similarity_topk(source, target, topk=10):
    count = get_similarity_count(source, target)
    val, idx = count.topk(dim=0, k=topk, sorted=True)
    return val, idx


def setup_parser():
    parser = configargparse.ArgParser(
        description='PyTorch PennTreeBank RNN/LSTM Language Model',
        default_config_files=['config/default.conf'],
    )

    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--data', type=str, default='./data/penn',
                        help='location of the data corpus')
    parser.add_argument('--vocab', type=str, default=None,
                        help='location of the vocabulary file, without which will use vocab of training corpus')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='initial weight decay')
    parser.add_argument('--lr-decay', type=float, default=2,
                        help='learning rate decay when no progress is observed on validation set')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--train', action='store_true',
                        help='set train mode, otherwise only evaluation is performed')
    parser.add_argument('--num-input-basis', type=int, default=0, # 0 will be converted to False
                        help='number of basis to decompose in input embedding matrix')
    parser.add_argument('--num-output-basis', type=int, default=0, # 0 will be converted to False
                        help='number of basis to decompose in output embedding matrix')
    parser.add_argument('--num-input-clusters', type=int, default=400,
                        help='number of clusters to use in each base for input matrix')
    parser.add_argument('--num-output-clusters', type=int, default=400,
                        help='number of clusters to use in each base for output matrix')

    return parser


def setup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('log/%s.log' % logger_name)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# Get the mask matrix of a batched input
def get_mask(lengths, cut_tail=0):
    assert lengths.min() >= cut_tail
    max_len = lengths.max()
    size = len(lengths)
    mask = lengths.new().byte().resize_(size, max_len).zero_()
    for i in range(size):
        mask[i][:lengths[i]-cut_tail].fill_(1)
    return Variable(mask)


def process_data(data_batch, cuda=False, sep_target=True):
    """A data pre-processing util which construct the input `Variable` for model

    Args:
        - data_batch: a batched data from `PaddedDataset`
        - cuda: indicates whether to put data into GPU
        - sep_target: return separated input and target if turned on

    Returns:
        - input: the input data batch
        - target: target data if `sep_target` is True, else a duplicated input
        - effective_length: the useful sentence length for loss computation <s> is ignored
        """

    batch_sentence, length = data_batch
    if cuda:
        batch_sentence = batch_sentence.cuda()
        length = length.cuda()

    # cut the padded sentence to max sentence length in this batch
    max_len = length.max()
    batch_sentence = batch_sentence[:, :max_len]

    # the useful sentence length for loss computation <s> is ignored
    effective_length = length - 1

    if sep_target:
        data = batch_sentence[:, :-1]
        target = batch_sentence[:, 1:]
    else:
        data = batch_sentence
        target = batch_sentence

    data = Variable(data.contiguous())
    target = Variable(target.contiguous(), requires_grad=False)

    return data, target, effective_length


