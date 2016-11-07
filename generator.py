from rnn_model import x_train, y_train, x_valid, y_valid,\
        l_out, calc_perplexity, BATCH_SIZE, REC_NUM_UNITS,\
        f_train, MODEL_WORD_LEN, num_epochs, no_decay_epochs, sh_lr,
        feature_extractor
import lasagne
import numpy as np
import cPickle
import time


with open('model.pickle', 'rb') as f:
    param_values = cPickle.load(f)
    lasagne.layers.set_all_param_values(l_out, param_values)

#TODO: define theano function that just generates output given input
#define function in extract features that can extract on the fly
# figure out how to generate new outputs given starter
#TODO: use model from https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
# where we do overlapping batches to predict the word following the batch
# refactor to make this easy
def generate(rapper):
    first_line = '(NRP: {})'.format(rapper)

    n_batches = starter[0].shape[0] // BATCH_SIZE
    hidden_states1 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in xrange(len(starter))]
    hidden_states2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in xrange(len(starter))]

    for i in range(n_batches):
        x_batch = [f[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for f in starter]
        y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        inputs = x_batch + [y_batch] + hidden_states1 + hidden_states2
        output = f_generate(*inputs)

    n_words_evaluated = (x[0].shape[0] - 1) / MODEL_WORD_LEN
    perplexity = np.exp(np.sum(l_cost) / n_words_evaluated)

    return perplexity
