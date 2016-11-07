from rnn_model import x_train, y_train, x_valid, y_valid,\
        l_out, calc_perplexity, BATCH_SIZE, REC_NUM_UNITS,\
        f_train, MODEL_WORD_LEN, num_epochs, no_decay_epochs, sh_lr
import lasagne
import numpy as np
import cPickle
import time





n_batches_train = x_train[0].shape[0] // BATCH_SIZE
for epoch in range(num_epochs):
    l_cost, l_norm, batch_time = [], [], time.time()

    # use zero as initial state
    hidden_states1 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in range(len(x_train))]
    hidden_states2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in range(len(x_train))]
    for i in range(n_batches_train):
        x_batch = [f[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for f in x_train]
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        _inputs = x_batch + [y_batch] + hidden_states1 + hidden_states2
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
