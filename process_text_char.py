from __future__ import division
import gzip
import numpy as np
import pandas as pd
import math
import sys


class TextProcessor(object):
    def __init__(self,
                 corpus_path,
                 filenames,
                 batch_size,
                 model_seq_len,
                 word_embedding_size,
                 gzipped=True):
        self.corpus_path = corpus_path
        self.model = {}
        self.special_symbols = {
            '<eos>': self.eos_vector(),
            '<eov>': self.eov_vector(),
            '<nrp>': self.nrp_vector(),
        }
        self.filenames = filenames
        self.gzipped = gzipped
        self.batch_size = batch_size
        self.model_seq_len = model_seq_len

        self.symbols = []
        self.char2sym = {}
        self.sym2char = {}
        self.vocab_length = 0
        self.max_word_len = 0

    @property
    def vector_dim(self):
        return self.max_word_len

    def load_file(self, filename):
        if self.gzipped:
            open_func = gzip.open
        else:
            open_func = open
        with open_func(filename, 'rb') as f:
            for line in f:
                symbols = self.process_line(line)
                self.symbols.extend(symbols)

    def process_line(self, line):
        line = line.lstrip().rstrip()
        words = line.split(" ")
        # TODO: Parse NRP
        if len(words) == 1 and words[0] == '':
            yield self.eov_vector()
        if len(words) == 1 and words[0] in self.special_symbols:
            yield self.special_symbols[words[0]]
        for i, word in enumerate(words):
            word_symbols = np.zeros(len(word),dtype=np.int32)
            if len(word) > self.max_word_len:
                self.max_word_len = len(word)
            for j, char in enumerate(word):
                if char not in self.char2sym:
                    cur_sym = self.vocab_length
                    self.char2sym[char] = cur_sym
                    self.sym2char[cur_sym] = char
                    word_symbols[j] = cur_sym
                    self.vocab_length += 1
                else:
                    word_symbols[j] = self.char2sym[char]
            yield word_symbols
        yield self.eos_vector()

    def eos_vector(self):
        return np.array([-2])

    def eov_vector(self):
        return np.array([-4])

    def nrp_vector(self):
        return np.array([-5])

    def calc_num_batches(self):
        x_resize =  \
            (len(self.symbols) // (self.batch_size*self.model_seq_len))*self.model_seq_len*self.batch_size
        self.n_samples = x_resize // self.model_seq_len
        self.n_batches = self.n_samples // self.batch_size
    def standardize_word_lengths(self):
        self.vectors = -1*np.ones((len(self.symbols), self.max_word_len))
        for i, s in enumerate(self.symbols):
            self.vectors[i,:s.shape[0]] = s

    def load_data(self):
        for f in self.filenames:
            self.load_file(f)
        self.standardize_word_lengths()
        self.calc_num_batches()


    def extract_ordered_data(self):
        x_in = self.vectors
        if x_in.shape[0] % (batch_size*model_seq_len) == 0:
            print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
                  "set to x_in = x_in[:-1]")
            x_in = x_in[:-1]

        x_resize =  \
            (x_in.shape[0] // (self.batch_size*self.model_seq_len))*self.model_seq_len*self.batch_size
        n_samples = x_resize // (self.model_seq_len)
        n_batches = n_samples // self.batch_size

        targets = x_in[1:x_resize+1].reshape(n_samples, self.model_seq_len, self.max_word_len)
        x_out = x_in[:x_resize].reshape(n_samples, self.model_seq_len, self.max_word_len)

        out = np.zeros(n_samples, dtype=int)

        for i in range(self.n_batches):
            val = range(i, self.n_batches*self.batch_size+i, self.n_batches)
            out[i*self.batch_size:(i+1)*self.batch_size] = val

        x_out = x_out[out]
        targets = targets[out]

        return x_out.astype('int32'), targets.astype('int32')

if __name__ == '__main__':
    #filenames = ['test_ints.txt']
    filenames = ['test_rap.txt']
    test_rap = range(100)
    test_rap = ' '.join([str(x) for x in test_rap])
    with open('test_ints.txt','wb') as f:
        f.write(test_rap)
    batch_size = 2
    model_seq_len = 3
    word_embedding_size = 1
    process = TextProcessor(None,
                 filenames,
                 batch_size,
                 model_seq_len,
                 word_embedding_size,
                 gzipped=False)
    process.load_data()
    # TODO: deal with word vectors of dim>1
    x, targets = process.extract_ordered_data()
    print x
    print targets

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
