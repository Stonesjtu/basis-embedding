## basis embedding

> code for Structured Word Embedding for Low Memory Neural Network Language Model

The code repo for basis embedding to reduce model size and memory consumption
This repo is built based on the pytorch/examples repo on github

### Parameters Introduction
basis embedding related arguments:
  - `--basis` <0>: number of basis to decompose the embedding matrix, 0 is normal mode
  - `--num_clusters`: number of clusters for all the vocabulary
  - `--load_input_embedding`: path of pre-trained embedding matrix for input embedding
  - `--load_output_embedding`: path of pre-trained embedding matrix for output embedding

misc options:
  - `-c` or `--config`: the path for configuration file, it will override arguments parser's
  default values and be overrided by command line options
  - `--train`: train or just evaluation existing model
  - `--dict <None>`: use vocabulary file if specified, otherwise use the words in train.txt

### examples

```bash
python main.py -c config/default.conf  # train a cross-entropy baseline
python main.py -c config/ptb_basis_tied.conf # basis embedding inited via tied embedding on ptb
```
During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluted against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  -c, --config PATH  preset configurations to load
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        humber of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
  ... more from previous basis embedding related parameters
```


### File Hierarchy

- main.py: the entry file, it parses the parameters, defines models
and feeds the data to model
- model.py: define the input embedding and LSTM layer
- basis_loss.py: It contains a basis linear module, taking inputs from LSTM hidden state and outputing loss value.
- basis/: core part of the basis embedding module
- utils.py: do product quantization for pre-trained embedding
- data.py: data pre-processing
- .th/.th.decoder: the pre-trained embedding matrix
- .conf: sample configuration files
