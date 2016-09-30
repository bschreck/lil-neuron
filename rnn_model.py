# use lasagne
"""
Examples that mimick the setup in http://arxiv.org/abs/1409.2329
except that we use GRU instead of a LSTM layers.
The example demonstrates:
    * How to setup GRU in Lasagne to predict a target for every position in a
      sequence.
    * How to setup Lasagne for language modelling tasks.
    * Fancy reordering of data to allow theano to process a large text
      corpus.
    * How to combine recurrent and feed-forward layers in the same Lasagne
      model.


STRATEGY:
    Each song includes a separate sub-network that's not recurrent with just metadata about the song
    so year, location, artist (embedded as a low-dimensional vector), featured artists, etc.

    Each song could addionally include the performing rapper as the start-of-verse symbol, using the same
    encoding as the artist and featured artists mentioned in the metadata

    Scrape from lyrics.wikia.com, after first scraping the artists to use from the top N rappers on
    last.fm, spotify, or echo nest, turn the artist's name into the format lyrics.wikia.com uses,
    go to the artist page there, and find the CSS class that lists all the songs (or albums) and urls
    Each song also includes an Artist: at the start of each verse if the artist is not the same as the
    artist who made the song, or who is a sub-artist (if it's a rap group)

    ALSO:
    strategy for word embeddings: since rappers already know what words mean in their usual context
    before even learning how to rap, I'll use Word2Vec to generate embeddings for the words in the corpus
    beforehand. Unknown words (mispellings or rap lingo) will be embedded by some linear combinations of
    the surrounding word vectors, maybe subtracting common words and adding uncommon words, as well as
    finding the words closest in spelling to the word in question
"""
from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import time
import pdb
import lasagne
import cPickle
from extract_features import RapFeatureExtractor

np.random.seed(1234)

BATCH_SIZE = 3#50                     # batch size
MODEL_WORD_LEN = 3#50                 # how many words to unroll
                                    # (some features may have multiple
                                    #  symbols per word)
TOL = 1e-6                          # numerial stability
INI = lasagne.init.Uniform(0.1)     # initial parameter values
EMBEDDING_SIZE = 20#400                # Embedding size
REC_NUM_UNITS = 20#400                 # number of LSTM units

dropout_frac = 0.1                  # optional recurrent dropout
lr = 2e-3                           # learning rate
decay = 2.0                         # decay factor
no_decay_epochs = 5                 # run this many epochs before first decay
max_grad_norm = 15                  # scale steps if norm is above this value
num_epochs = 1000                   # Number of epochs to run


# Theano symbolic vars
sym_y = T.imatrix()




# BUILDING THE MODEL
def build_rnn(hid1_init_sym, hid2_init_sym, model_seq_len, word_vector_size):
# Model structure:
#
#    embedding --> GRU1 --> GRU2 --> output network --> predictions
    l_inp = lasagne.layers.InputLayer((BATCH_SIZE, model_seq_len, word_vector_size))
    print("l_inp:", l_inp.output_shape)

    # TODO: this should be a layer that keeps the number of dimensions,
    # and just alters the last dimension into EMBEDDING_SIZE
    # maybe reshape input layer into 2D, and then add embedding layer
    l_emb = lasagne.layers.NiNLayer(
        l_inp,
        num_units=EMBEDDING_SIZE,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=INI)
    # TODO: include bias?

    print("l_emb:", l_emb.output_shape)

    l_drp0 = lasagne.layers.DropoutLayer(l_emb, p=dropout_frac)

    def create_gate():
        return lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None)

# first GRU layer
    l_rec1 = lasagne.layers.GRULayer(
        l_drp0,
        num_units=REC_NUM_UNITS,
        resetgate=create_gate(),
        updategate=create_gate(),
        hidden_update=create_gate(),
        learn_init=False,
        hid_init=hid1_init_sym)
    print('l_rec1:', l_rec1.output_shape)

    l_drp1 = lasagne.layers.DropoutLayer(l_rec1, p=dropout_frac)

# Second GRU layer
    l_rec2 = lasagne.layers.GRULayer(
        l_drp1,
        num_units=REC_NUM_UNITS,
        resetgate=create_gate(),
        updategate=create_gate(),
        hidden_update=create_gate(),
        learn_init=False,
        hid_init=hid2_init_sym)

    l_drp2 = lasagne.layers.DropoutLayer(l_rec2, p=dropout_frac)
    return [l_inp, l_emb, l_drp0, l_rec1, l_drp1, l_rec2, l_drp2]

train_filenames = ['data/test_rap.txt']
valid_filenames = ['data/test_rap.txt']
gzipped = False
feature_extractor = RapFeatureExtractor(train_filenames = train_filenames,
                                        valid_filenames = valid_filenames,
                                        batch_size = BATCH_SIZE,
                                        model_word_len = MODEL_WORD_LEN,
                                        gzipped = gzipped)
x_train, y_train, x_valid, y_valid = feature_extractor.extract()
# TODO: feature_extractor should save vocab_length, other stuff in pickle file
vocab_size = feature_extractor.vocab_length
char2sym = feature_extractor.char2sym
sym2char = feature_extractor.sym2char


features = feature_extractor.feature_set
final_layers = []
rec_layers = []
input_dict = {}
inputs = {'x':[],'y':[sym_y], 'h':[]}
total_model_len = 0
for f in features:
    feature_vector_size = f.vector_dim
    total_model_len += REC_NUM_UNITS

    input_X = T.imatrix()
    hid1_init_sym = T.matrix()
    hid2_init_sym = T.matrix()
    inputs['x'].append(input_X)
    inputs['h'].extend([hid1_init_sym, hid2_init_sym])

    [l_inp, l_emb, l_drp0, l_rec1, l_drp1, l_rec2, l_drp2] = \
        build_rnn(hid1_init_sym, hid2_init_sym,
                  MODEL_WORD_LEN, feature_vector_size)
    input_dict[l_inp] = input_X

    final_layers.append(l_drp2)
    rec_layers.extend([l_rec1, l_rec2])

inputs = inputs['x'] + inputs['y'] + inputs['h']

concat_layer = lasagne.layers.ConcatLayer(final_layers, axis=1)

# by reshaping we can combine feed-forward and recurrent layers in the
# same Lasagne model.
print("total model len:", total_model_len)
print("rec num units:", REC_NUM_UNITS)
print(concat_layer.output_shape)
l_shp = lasagne.layers.ReshapeLayer(concat_layer,
                                    (BATCH_SIZE * total_model_len, REC_NUM_UNITS))
print(l_shp.output_shape)
print("VOCAB SIZE:", vocab_size)
l_out = lasagne.layers.DenseLayer(l_shp,
                                  num_units=vocab_size,
                                  nonlinearity=lasagne.nonlinearities.softmax)
print("l_out:", l_out.output_shape)
l_out = lasagne.layers.ReshapeLayer(l_out,
                                    (BATCH_SIZE,
                                     total_model_len,
                                     vocab_size))


def calc_cross_ent(net_output, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (BATCH_SIZE * MODEL_WORD_LEN, vocab_size))
    preds += TOL  # add constant for numerical stability
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)
    return cost

# Note the use of deterministic keyword to disable dropout during evaluation.
train_out_layers = lasagne.layers.get_output(
        [l_out] + rec_layers, input_dict, deterministic=False)
train_out = train_out_layers[0]


# after we have called get_ouput then the layers will have reference to
# their output values. We need to keep track of the output values for both
# training and evaluation and for each hidden layer because we want to
# initialze each batch with the last hidden values from the previous batch.
hidden_states_train = train_out_layers[1:]

eval_out_layers = lasagne.layers.get_output(
    [l_out] + rec_layers, input_dict, deterministic=True)
eval_out = eval_out_layers[0]
hidden_states_eval = eval_out_layers[1:]

cost_train = T.mean(calc_cross_ent(train_out, sym_y))
cost_eval = T.mean(calc_cross_ent(eval_out, sym_y))

# Get list of all trainable parameters in the network.
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# Calculate gradients w.r.t cost function. Note that we scale the cost with
# MODEL_WORD_LEN. This is to be consistent with
# https://github.com/wojzaremba/lstm . The scaling is due to difference
# between torch and theano. We could have also scaled the learning rate, and
# also rescaled the norm constraint.
all_grads = T.grad(cost_train * MODEL_WORD_LEN, all_params)

all_grads = [T.clip(g, -5, 5) for g in all_grads]

# With the gradients for each parameter we can calculate update rules for each
# parameter. Lasagne implements a number of update rules, here we'll use
# sgd and a total_norm_constraint.
all_grads, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_grad_norm, return_norm=True)


# Use shared variable for learning rate. Allows us to change the learning rate
# during training.
sh_lr = theano.shared(lasagne.utils.floatX(lr))
updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=sh_lr)

# Define evaluation function. This graph disables dropout.
print("compiling f_eval...")
f_eval = theano.function(inputs,
                         [cost_eval]+
                          [t for t in hidden_states_eval])

# define training function. This graph has dropout enabled.
# The update arg specifies that the parameters should be updated using the
# update rules.
print("compiling f_train...")
# used to be [t[:-1] for t in hidden_states_train]
f_train = theano.function(inputs,
                          [cost_train, norm] +
                          [t for t in hidden_states_train],
                          updates=updates)


def calc_perplexity(x, y):
    """
    Helper function to evaluate perplexity.
    Perplexity is the inverse probability of the test set, normalized by the
    number of words.
    See: https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
    This function is largely based on the perplexity calcualtion from
    https://github.com/wojzaremba/lstm/
    """

    n_batches = x[0].shape[0] // BATCH_SIZE
    l_cost = []

    hidden_states = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in xrange(len(x))]

    for i in range(n_batches):
        x_batch = [f[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for f in x]
        y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        inputs = x_batch + [y_batch] + hidden_states
        output = f_eval(*inputs)
        cost = output[0]
        l_cost.append(cost)

    n_words_evaluated = (x[0].shape[0] - 1) / MODEL_WORD_LEN
    perplexity = np.exp(np.sum(l_cost) / n_words_evaluated)

    return perplexity

n_batches_train = x_train[0].shape[0] // BATCH_SIZE
for epoch in range(num_epochs):
    l_cost, l_norm, batch_time = [], [], time.time()

    # use zero as initial state
    hidden_states = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in range(len(x_train))]
    for i in range(n_batches_train):
        x_batch = [f[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for f in x_train]
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        _inputs = x_batch + [y_batch] + hidden_states
        output = f_train(*_inputs)
        cost = output[0]
        norm = output[1]
        l_cost.append(cost)
        l_norm.append(norm)
    with open('model.pickle', 'wb') as f:
        cPickle.dump(lasagne.layers.get_all_param_values(l_out), f,
                     cPickle.HIGHEST_PROTOCOL)

    if epoch > (no_decay_epochs - 1):
        current_lr = sh_lr.get_value()
        sh_lr.set_value(lasagne.utils.floatX(current_lr / float(decay)))

    elapsed = time.time() - batch_time
    words_per_second = float(BATCH_SIZE*(MODEL_WORD_LEN)*len(l_cost)) / elapsed
    n_words_evaluated = (x_train[0].shape[0] - 1) / MODEL_WORD_LEN
    perplexity_valid = calc_perplexity(x_valid, y_valid)
    perplexity_train = np.exp(np.sum(l_cost) / n_words_evaluated)
    print("Epoch           :", epoch)
    print("Perplexity Train:", perplexity_train)
    print("Perplexity valid:", perplexity_valid)
    print("Words per second:", words_per_second)
    l_cost = []
    batch_time = 0
