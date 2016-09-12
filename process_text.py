from __future__ import division
import gzip
from gensim.models import Word2Vec
import numpy as np
import math
import sys

# TODO: initially I wasn't going to load everything in memory, but it seems
# that's the best way to do this

class TextProcessor(object):
    def __init__(self,
                 corpus_path,
                 filenames,
                 batch_size,
                 model_seq_len):
        self.corpus_path = corpus_path
        self.model = Word2Vec.load_word2vec_format(self.corpus_path, binary=True)
        self.special_symbol_vecs = {
            '<eos>': self.eos_vector(),
            '<eov>': self.eov_vector(),
            '<eop>': self.eop_vector(),
            '<nrp>': self.nrp_vector(),
        }
        self.filenames = filenames
        self.batch_size = batch_size
        self.model_seq_len = model_seq_len

        self.symbols = []
        self.vocab = set()
        self.sym2word = {}
        self.vocab_length = 0

    def load_file(self, filename):
        with gzip.open(filename, 'rb') as f:
            for line in f:
                symbols = self.process_line(line)
                self.symbols.extend(symbols)

    def process_line(self, line):
        line = line.lstrip().rstrip()
        words = line.split(" ")
        words.append('<eos>')
        # TODO: parse <eov>, <eop>, <nrp>
        for i, word in enumerate(words):
            if word not in self.vocab:
                self.vocab.add(word)
                self.sym2word[self.vocab_length] = word
                yield self.vocab_length
                self.vocab_length += 1
            else:
                yield self.sym2word[word]

    def word2vec(self, i, sym):
        word = self.sym2word[sym]
        if word in self.model:
            vec = self.model[word]
        elif word in self.special_symbol_vecs:
            vec = self.special_symbol_vecs[word]
        else:
            vec = self.unknown_word(i)
        yield vec

    def unknown_word(self, i):
        # TODO: figure out scheme for unknown words
        pass

    def eos_vector(self):
        # TODO: figure out how to initialize this
        pass

    def eop_vector(self):
        # TODO: figure out how to initialize this
        pass

    def eov_vector(self):
        # TODO: figure out how to initialize this
        pass

    def nrp_vector(self):
        # TODO: figure out how to initialize this
        pass

    def calc_num_batches(self):
        x_resize =  \
            (x_in.shape[0] // (self.batch_size*self.model_seq_len))*self.model_seq_len*self.batch_size
        self.n_samples = x_resize // self.model_seq_len
        self.n_batches = self.n_samples // self.batch_size

    def load_data(self):
        for f in self.filenames:
            self.load_file(f)
        self.symbols = np.array(self.symbols)
        self.calc_num_batches()

    def extract_batch(self):
        look_ahead_index = self.model_seq_len
        for i in range(self.batch_size - 1):
            look_ahead_index += (self.num_batches*self.model_seq_len)
        batches_to_keep = int(math.ceil(look_ahead_index / (self.batch_size*self.model_seq_len)))

        rows_to_pull = np.zeros(self.batch_size, dtype=int)
        for i in xrange(1, self.batch_size):
            rows_to_pull[i] = rows_to_pull[i-1] + self.batch_size + 1


        prev_flat_batches = np.zeros((batches_to_keep*
                                      self.batch_size,
                                      self.model_seq_len,
                                      self.word_embedding_size))


        total_preloaded = batches_to_keep * self.batch_size * self.model_seq_len
        i = 0
        for b in xrange(batches_to_keep * self.batch_size):
            for j in xrange(self.model_seq_len):
                sym = self.symbols[i]
                vec = self.word2vec(i, sym)
                prev_flat_batches[b, j, :] = vec
                i += 1
        for i, sym in enumerate(self.symbols[total_preloaded:]):
            vec = self.word2vec(i + total_preloaded, sym)
            prev_flat_batches[-1, j, :] = vec

            x_out = prev_flat_batches[rows_to_pull]
            prev_flat_batches = np.roll(prev_flat_batches, -1, axis=0)
            yield x_out

# def reorder(x_in, batch_size, model_seq_len):
    # """
    # Rearranges data set so batches process sequential data.
    # If we have the dataset:
    # x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # and the batch size is 2 and the model_seq_len is 3. Then the dataset is
    # reordered such that:
                   # Batch 1    Batch 2
                 # ------------------------
    # batch pos 1  [1, 2, 3]   [4, 5, 6]
    # batch pos 2  [7, 8, 9]   [10, 11, 12]
    # This ensures that we use the last hidden state of batch 1 to initialize
    # batch 2.
    # Also creates targets. In language modelling the target is to predict the
    # next word in the sequence.
    # Parameters
    # ----------
    # x_in : 1D numpy.array
    # batch_size : int
    # model_seq_len : int
        # number of steps the model is unrolled
    # Returns
    # -------
    # reordered x_in and reordered targets. Targets are shifted version of x_in.
    # """
    # if x_in.ndim != 1:
        # raise ValueError("Data must be 1D, was", x_in.ndim)

    # if x_in.shape[0] % (batch_size*model_seq_len) == 0:
        # print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
              # "set to x_in = x_in[:-1]")
        # x_in = x_in[:-1]

    # x_resize =  \
        # (x_in.shape[0] // (batch_size*model_seq_len))*model_seq_len*batch_size
    # n_samples = x_resize // (model_seq_len)
    # n_batches = n_samples // batch_size
    # print n_samples
    # print n_batches

    # targets = x_in[1:x_resize+1].reshape(n_samples, model_seq_len)
    # x_out = x_in[:x_resize].reshape(n_samples, model_seq_len)
    # print x_out

    # out = np.zeros(n_samples, dtype=int)

    # for i in range(n_batches):
        # val = range(i, n_batches*batch_size+i, n_batches)
        # out[i*batch_size:(i+1)*batch_size] = val

    # x_out = x_out[out]
    # targets = targets[out]

    # return x_out.astype('int32'), targets.astype('int32')


# def test(data, batch_size, model_seq_len, num_batches):
        # #TODO: test this, and figure out how to get num_batches

        # look_ahead_index = model_seq_len
        # for i in range(batch_size - 1):
            # look_ahead_index += (num_batches*model_seq_len)
        # batches_to_keep = int(math.ceil(look_ahead_index / (batch_size*model_seq_len)))

        # rows_to_pull = np.zeros(batch_size, dtype=int)
        # for i in xrange(1, batch_size):
            # rows_to_pull[i] = rows_to_pull[i-1] + batch_size + 1


        # prev_flat_batches = np.zeros((batches_to_keep*
                                      # batch_size,
                                      # model_seq_len))

        # start = True
        # while True:
            # if start:
                # for b in xrange(batches_to_keep * batch_size):
                    # for j in xrange(model_seq_len):
                        # try:
                            # vec = data.next()
                        # except StopIteration:
                            # raise StopIteration
                        # prev_flat_batches[b, j] = vec
                # start = False
            # else:
                # for j in xrange(model_seq_len):
                    # vec = data.next()
                    # prev_flat_batches[-1, j] = vec

            # x_out = prev_flat_batches[rows_to_pull]
            # prev_flat_batches = np.roll(prev_flat_batches, -1, axis=0)
            # yield x_out


if __name__ == '__main__':
    x_in = np.arange(25)
    def data(x_in):
        for i in x_in:
            yield i
    x_gen = test(data(x_in), 3, 2, 4)
    for i in x_gen:
        print i
    # x_out = reorder(x_in, 3, 1)
    # print x_out

 # 0,  1],
# [ 8,  9],
# [16, 17],
# [ 2,  3],
# [10, 11],
# [18, 19],
# [ 4,  5],
# [12, 13],
# [20, 21],
# [ 6,  7],
# [14, 15],
# [22, 23]
