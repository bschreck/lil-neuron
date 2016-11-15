"""
Pronunciation features:
    - phonemes as stresses
    - raw phonemes, no stresses
    - raw phonemes, broken up into groups separated by stresses
    - raw phonemes, broken up into groups separated by words
    - raw phonemes, broken up into groups separated by phrases or lines
    - rhyme scheme as categorical variables (mark each rhyme with same category)
"""
import pronouncing
import numpy as np
import cPickle as pickle
import os
import pdb
import gzip
from features import PhonemeFeature, WordFeature, StressFeature
from features import RawStresses, RawPhonemes, RawWordsAsChars
from features import RapVectorFeature
from generate_lyric_files import all_filenames
import tensorflow as tf
import inflect
import re
import time

#TODO for pronouncing
# how to deal with non-dictionary words?


class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
    phone2int = {p: i for i, p in enumerate(all_phones)}

    def __init__(self,
                 train_filenames=None,
                 valid_filenames=None,
                 batch_size=3,
                 model_word_len=3,
                 max_rappers_per_verse=3,
                 word2sym={},
                 sym2word={},
                 char2sym={},
                 sym2char={},
                 char_vocab_length=0,
                 vocab_length=0,
                 sequence_feature_set=None,
                 static_feature_set=None,
                 gzipped=True,
                 pickle_file=None,
                 rapper_vector_file='rapper_matrix.p'):
        self.train_filenames = train_filenames
        self.valid_filenames = valid_filenames
        self.pck_data_file = pickle_file
        if self.pck_data_file is None:
            self.pck_data_file = os.path.join("data", "formatted_data.p")
        self.gzipped = gzipped

        self.batch_size = batch_size
        self.model_word_len = model_word_len
        self.rapper_vectors = pickle.load(open(rapper_vector_file))
        for key, vec in self.rapper_vectors.iteritems():
            self.rapper_vectors[key] = vec.astype(np.int32)
        self.len_rapper_vector = self.rapper_vectors.values()[0].shape[0]
        self.char2sym = char2sym
        self.sym2char = sym2char
        self.word2sym = word2sym
        self.sym2word = sym2word
        self.char_vocab_length = char_vocab_length
        self.vocab_length = vocab_length
        self.max_word_len = 1
        self.max_prons_per_word = 1
        self.max_phones_per_word = 1
        self.max_nrps = max_rappers_per_verse

        inflect_engine = inflect.engine()
        self.inflect_num_to_words = inflect_engine.number_to_words

        self.special_symbols = {
        }
        special_symbols = ['<eos>', '<eov>', '<nrp>']
        for s in special_symbols:
            self.word2sym[s] = self.vocab_length
            self.sym2word[self.vocab_length] = s
            self.special_symbols[s] = self.vocab_length
            self.vocab_length += 1

        self.rapper2sym = {}
        self.sym2rapper = {}

        self.train_word_syms = []
        self.train_char_syms = []
        self.train_phone_syms = []
        self.train_stress_syms = []
        self.train_rap_vecs = []
        self.valid_word_syms = []
        self.valid_char_syms = []
        self.valid_phone_syms = []
        self.valid_stress_syms = []
        self.valid_rap_vecs = []

        if sequence_feature_set:
            self.sequence_feature_set = sequence_feature_set
        else:
            self.sequence_feature_set = [RawWordsAsChars(),
                                         RawStresses(),
                                         RawPhonemes()]
        if static_feature_set:
            self.static_feature_set = static_feature_set
        else:
            self.static_feature_set = [RapVectorFeature()]

    def _add_rappers(self, *rappers):
        # TODO: also include context feature for pronunciation
        # of rapper, and characters in the word
        rapper_syms = []
        rapper_vectors = []
        for rapper in rappers:
            if rapper not in self.rapper2sym:
                cur_sym = len(self.rapper2sym)
                self.rapper2sym[rapper] = cur_sym
                self.sym2rapper[cur_sym] = rapper
            else:
                cur_sym = self.rapper2sym[rapper]
            if rapper in self.rapper_vectors:
                rapper_vector = self.rapper_vectors[rapper]
                rapper_vectors.append(rapper_vector)
            else:
                rapper_vectors.append(np.zeros(self.len_rapper_vector))
            rapper_syms.append(cur_sym)
        if len(rapper_vectors) > self.max_nrps:
            #self.max_nrps = len(rapper_vectors)
            rapper_syms = rapper_syms[:self.max_nrps]
            rapper_vectors = rapper_vectors[:self.max_nrps]
        return rapper_syms, rapper_vectors

    def _nrp_line(self, line):
        if line.startswith('(NRP:'):
            words = line.split('(NRP:')
            rappers = [w.replace(")","").strip() for w in words
                       if w]
            return self._add_rappers(*rappers)
            # words = words[1:]
            # first_rapper = words[0].split('(', 1)[1]
            # if len(words) > 1:
                # last_rapper = words[-1][:-1]
            # else:
                # first_rapper = first_rapper[:-1]
                # return self._add_rappers(first_rapper)
            # if len(words) > 2:
                # other_rappers = words[1:-1]
            # else:
                # other_rappers = []
            # rappers = [first_rapper] + other_rappers + [last_rapper]
            # return self._add_rappers(*rappers)
        return None, None

    # def _process_line(self, line):
        # line = line.lstrip().rstrip()
        # new_rappers, rapper_vectors = self._nrp_line(line)
        # if new_rappers is not None:
            # self.current_rapper_vecs = rapper_vectors
            # word_feat = [self.special_symbols['<nrp>']] + new_rappers + [self.special_symbols['<eos>']]
            # char_feat = [np.array(word_feat)]
            # other_feats = [char_feat]
            # return [word_feat, char_feat, other_feats, other_feats, [self.current_rapper_vecs]]

        # words = line.split(" ")

        # if len(words) == 1 and words[0] == '':
            # word_feat = [self.special_symbols['<eov>']]
            # char_feat = [np.array(word_feat)]
            # other_feats = [char_feat]
            # return [word_feat, char_feat, other_feats, other_feats, [self.current_rapper_vecs]]

        # if len(words) == 1 and words[0] in self.special_symbols:
            # word_feat = [self.special_symbols[words[0]]]
            # char_feat = [np.array(word_feat)]
            # other_feats = [char_feat]
            # return [word_feat, char_feat, other_feats, other_feats, [self.current_rapper_vecs]]

        # line_word_syms = []
        # line_char_syms = []
        # line_phones = []
        # line_stresses = []
        # for i, word in enumerate(words):
            # if word not in self.word2sym:
                # cur_word = self.vocab_length
                # self.word2sym[word] = cur_word
                # self.sym2word[cur_word] = word
                # self.vocab_length += 1
            # line_word_syms.append(self.word2sym[word])
            # word_symbols = np.zeros(len(word), dtype=np.int32)
            # word_phones, word_stresses = self._extract_phones(word)
            # if len(word) > self.max_word_len:
                # self.max_word_len = len(word)
            # for j, char in enumerate(word):
                # if char not in self.char2sym:
                    # cur_sym = self.char_vocab_length
                    # self.char2sym[char] = cur_sym
                    # self.sym2char[cur_sym] = char
                    # word_symbols[j] = cur_sym
                    # self.char_vocab_length += 1
                # else:
                    # word_symbols[j] = self.char2sym[char]
            # line_char_syms.append(word_symbols)
            # line_phones.append(word_phones)
            # line_stresses.append(word_stresses)

        # line_word_syms.append(self.special_symbols['<eos>'])
        # line_char_syms.append(np.array([self.special_symbols['<eos>']]))
        # line_phones.append([np.array([self.special_symbols['<eos>']])])
        # line_stresses.append([np.array([self.special_symbols['<eos>']])])
        # rap_vecs = [self.current_rapper_vecs] * len(line_word_syms)
        # return line_word_syms, line_char_syms, line_phones, line_stresses, rap_vecs

    def _process_line_new(self, line):
        line = line.lstrip().rstrip()
        new_rappers, rapper_vectors = self._nrp_line(line)
        if new_rappers is not None:
            word_feat = [self.special_symbols['<nrp>']] + new_rappers + [self.special_symbols['<eos>']]
            if len(word_feat) > self.max_word_len:
                self.max_word_len = len(word_feat)
            if len(word_feat) > self.max_phones_per_word:
                self.max_phones_per_word = len(word_feat)
            char_feat = [np.array(word_feat)]
            other_feats = [char_feat]
            return [word_feat, char_feat, other_feats, other_feats, rapper_vectors, False]

        words = line.split(" ")

        if len(words) == 1 and words[0] == '':
            word_feat = [self.special_symbols['<eov>']]
            char_feat = [np.array(word_feat)]
            other_feats = [char_feat]
            return [word_feat, char_feat, other_feats, other_feats, None, True]

        if len(words) == 1 and words[0] in self.special_symbols:
            word_feat = [self.special_symbols[words[0]]]
            char_feat = [np.array(word_feat)]
            other_feats = [char_feat]
            eov = words[0] == '<eov>'
            return [word_feat, char_feat, other_feats, other_feats, None, eov]

        line_word_syms = []
        line_char_syms = []
        line_phones = []
        line_stresses = []
        for i, word in enumerate(words):
            dictwords, syms = self._get_dict_words(word)
            for i, dword in enumerate(dictwords):
                cur_sym = syms[i]
                # TODO: use precomputed syms, make sure to preinitialize vocab_length
                # if dword not in self.word2sym:
                    # cur_word = self.vocab_length
                    # self.word2sym[dword] = cur_word
                    # self.sym2word[cur_word] = dword
                    # self.vocab_length += 1
                line_word_syms.append(cur_sym)#self.word2sym[dword])
                word_symbols = np.zeros(len(dword), dtype=np.int32)
                word_phones, word_stresses = self._extract_phones(dword)
                if len(dword) > self.max_word_len:
                    self.max_word_len = len(dword)
                for j, char in enumerate(dword):
                    if char not in self.char2sym:
                        cur_sym = self.char_vocab_length
                        self.char2sym[char] = cur_sym
                        self.sym2char[cur_sym] = char
                        word_symbols[j] = cur_sym
                        self.char_vocab_length += 1
                    else:
                        word_symbols[j] = self.char2sym[char]
                line_char_syms.append(word_symbols)
                line_phones.append(word_phones)
                line_stresses.append(word_stresses)

        line_word_syms.append(self.special_symbols['<eos>'])
        line_char_syms.append(np.array([self.special_symbols['<eos>']]))
        line_phones.append([np.array([self.special_symbols['<eos>']])])
        line_stresses.append([np.array([self.special_symbols['<eos>']])])
        return line_word_syms, line_char_syms, line_phones, line_stresses, None, False

    def _get_dict_words(self, word):
        replace_punctuation = string.maketrans(string.punctuation, trans)
        trans = word.lower().translate(replace_punctuation).split()
        dwords = []
        syms = []
        for w in trans:
            int_dwords = self.word_to_dwordint[w]
            syms.extend(int_dwords)
            [dwords.append(self.int_to_dword[i]) for i in int_dwords]
        return dwords, syms


    # def _load_file(self, fname, valid=False):
        # if self.gzipped:
            # open_func = gzip.open
        # else:
            # open_func = open
        # all_word_symbols = []
        # all_char_symbols = []
        # all_phones = []
        # all_stresses = []
        # all_rap_vecs = []
        # with open_func(fname, 'rb') as f:
            # for line in f:
                # word_symbols, char_symbols, phones, stresses, rap_vecs = self._process_line(line)
                # all_word_symbols.extend(word_symbols)
                # all_char_symbols.extend(char_symbols)
                # all_phones.extend(phones)
                # all_stresses.extend(stresses)
                # all_rap_vecs.extend(rap_vecs)
        # if valid:
            # self.valid_word_syms.extend(all_word_symbols)
            # self.valid_char_syms.extend(all_char_symbols)
            # self.valid_phone_syms.extend(all_phones)
            # self.valid_stress_syms.extend(all_stresses)
            # self.valid_rap_vecs.extend(all_rap_vecs)
        # else:
            # self.train_word_syms.extend(all_word_symbols)
            # self.train_char_syms.extend(all_char_symbols)
            # self.train_phone_syms.extend(all_phones)
            # self.train_stress_syms.extend(all_stresses)
            # self.train_rap_vecs.extend(all_rap_vecs)

    def _load_file_verses(self, fname, valid=False):
        if self.gzipped:
            open_func = gzip.open
        else:
            open_func = open
        verse_word_symbols = []
        verse_char_symbols = []
        verse_phones = []
        verse_stresses = []
        original_rap_vecs = None
        verse_rap_vecs = None
        with open_func(fname, 'rb') as f:
            for line in f:
                # remove \n
                line = line[:-1]
                word_symbols, char_symbols, phones, stresses, rap_vecs, eov = self._process_line_new(line)

                verse_word_symbols.extend(word_symbols)
                verse_char_symbols.extend(char_symbols)
                verse_phones.extend(phones)
                verse_stresses.extend(phones)
                if original_rap_vecs is None:
                    original_rap_vecs = rap_vecs
                if verse_rap_vecs is None and rap_vecs:
                    verse_rap_vecs = rap_vecs
                else:
                    verse_rap_vecs = original_rap_vecs
                if eov:
                    yield (verse_word_symbols,
                           verse_char_symbols,
                           verse_phones,
                           verse_stresses,
                           verse_rap_vecs)
                    verse_word_symbols = []
                    verse_char_symbols = []
                    verse_phones = []
                    verse_stresses = []
                    verse_rap_vecs = None

    # def _save_data_to_pickle(self):
        # data = [self.train_word_syms, self.valid_word_syms,
                # self.train_char_syms, self.valid_char_syms,
                # self.train_phone_syms, self.valid_phone_syms,
                # self.train_stress_syms, self.valid_stress_syms,
                # self.train_rap_vecs, self.valid_rap_vecs]
        # metadata = [self.char_vocab_length, self.char2sym, self.sym2char,
                    # self.vocab_length, self.word2sym, self.sym2word,
                    # self.rapper2sym, self.sym2rapper]
        # with open(self.pck_data_file, 'wb') as f:
            # pickle.dump([data, metadata], f)

    # def _load_data_from_pickle(self):
        # if os.path.isfile(self.pck_data_file):
            # with open(self.pck_data_file, 'rb') as f:
                # return pickle.load(f)
        # return None

    # def _load_data(self):
        # data = self._load_data_from_pickle()
        # if data is None or len(data) != 2:
            # for f in self.train_filenames:
                # self._load_file(f)
            # for f in self.valid_filenames:
                # self._load_file(f, valid=True)
            # self._standardize_word_lengths()
            # self._save_data_to_pickle()
        # else:
            # data, metadata = data
            # self.char_vocab_length, self.char2sym, self.sym2char = metadata[:3]
            # self.vocab_length, self.word2sym, self.sym2word = metadata[3:6]
            # self.rapper2sym, self.sym2rapper = metadata[6:8]
            # self.train_word_syms, self.valid_word_syms = data[:2]
            # self.train_char_syms, self.valid_char_syms = data[2:4]
            # self.train_phone_syms, self.valid_phone_syms = data[4:6]
            # self.train_stress_syms, self.valid_stress_syms = data[6:8]
            # self.train_rap_vecs, self.valid_rap_vecs = data[8:10]
        # self.x_resize_train, self.n_samples_train, self.n_batches_train =\
            # self._calc_num_batches(len(self.train_char_syms))
        # self.x_resize_valid, self.n_samples_valid, self.n_batches_valid =\
            # self._calc_num_batches(len(self.valid_char_syms))

    # def _calc_num_batches(self, len_symbols):
        # if len_symbols % (self.batch_size * self.model_word_len) == 0:
            # print(" x_in.shape[0] % (batch_size*model_word_len) == 0 -> x_in is "
                  # "set to x_in = x_in[:-1]")
            # len_symbols -= 1
        # div = len_symbols // (self.batch_size * self.model_word_len)
        # x_resize = div * self.model_word_len * self.batch_size
        # n_samples = x_resize // self.model_word_len
        # n_batches = n_samples // self.batch_size
        # return x_resize, n_samples, n_batches

    # def _standardize_word_lengths(self):
        # self.train_word_syms = np.array(self.train_word_syms)
        # self.valid_word_syms = np.array(self.valid_word_syms)
        # new_word_vectors = []
        # for word_data in [self.train_char_syms, self.valid_char_syms]:
            # vectors = -1 * np.ones((len(word_data), self.max_word_len))
            # for i, s in enumerate(word_data):
                # vectors[i, :s.shape[0]] = s
            # new_word_vectors.append(vectors)
        # self.train_char_syms, self.valid_char_syms = new_word_vectors

        # word_size = self.max_phones_per_word * self.max_prons_per_word
        # phone_sz = self.max_phones_per_word
        # new_phone_vectors = []
        # for phone_data in [self.train_phone_syms, self.valid_phone_syms,
                           # self.train_stress_syms, self.valid_stress_syms]:
            # vectors = -1 * np.ones((len(phone_data), word_size))
            # for i, s in enumerate(phone_data):
                # for j, pron in enumerate(s):
                    # vectors[i, j * phone_sz: j * phone_sz + len(pron)] = pron
            # new_phone_vectors.append(vectors)
        # self.train_phone_syms, self.valid_phone_syms = new_phone_vectors[:2]
        # self.train_stress_syms, self.valid_stress_syms = new_phone_vectors[2:]

        # self.train_rap_vecs = np.array(self.train_rap_vecs)
        # self.valid_rap_vecs = np.array(self.valid_rap_vecs)
        # new_rap_vectors = []
        # vecsize = self.len_rapper_vector
        # for rapper_data in [self.train_rap_vecs, self.valid_rap_vecs]:
            # vectors = -1 * np.ones((len(rapper_data), vecsize * self.max_nrps))
            # for i, s in enumerate(rapper_data):
                # for j, rap_vec in enumerate(s):
                    # vectors[i, j * vecsize: (j+1) * vecsize] = rap_vec
            # new_rap_vectors.append(vectors)
        # self.train_rap_vecs, self.valid_rap_vecs = new_rap_vectors

    # def _reorder_data(self, x_in, n_batches, n_samples, x_resize, vector_dim):
        # if vector_dim > 0:
            # args = [n_samples, self.model_word_len, vector_dim]
        # else:
            # args = [n_samples, self.model_word_len]
        # targets = x_in[1: x_resize + 1].reshape(*args)

        # x_out = x_in[:x_resize].reshape(*args)

        # out = np.zeros(n_samples, dtype=int)

        # for i in range(n_batches):
            # val = range(i, n_batches * self.batch_size + i, n_batches)
            # out[i * self.batch_size: (i + 1) * self.batch_size] = val

        # x_out = x_out[out]
        # targets = targets[out]

        # return x_out.astype('int32'), targets.astype('int32')

    # def extract(self):
        # self._load_data()
        # unordered_raw_t = self.train_word_syms
        # _, y_t = self._reorder_data(unordered_raw_t,
                                    # self.n_batches_train,
                                    # self.n_samples_train,
                                    # self.x_resize_train,
                                    # 0)
        # unordered_raw_v = self.valid_word_syms
        # _, y_v = self._reorder_data(unordered_raw_v,
                                    # self.n_batches_train,
                                    # self.n_samples_train,
                                    # self.x_resize_train,
                                    # 0)

        # x_train_sequence = []
        # x_valid_sequence = []
        # for f in self.sequence_feature_set:
            # if isinstance(f, PhonemeFeature):
                # data = [self.train_phone_syms, self.valid_phone_syms]
            # elif isinstance(f, StressFeature):
                # data = [self.train_stress_syms, self.valid_stress_syms]
            # elif isinstance(f, WordFeature):
                # data = [self.train_char_syms, self.valid_char_syms]
            # else:
                # raise ValueError("Unkown feature type: {}".format(f.__class__))
            # unordered_t, unordered_v = f.extract(*data)
            # x_t, _ = self._reorder_data(unordered_t,
                                          # self.n_batches_train,
                                          # self.n_samples_train,
                                          # self.x_resize_train,
                                          # f.vector_dim)
            # x_v, _ = self._reorder_data(unordered_v,
                                          # self.n_batches_valid,
                                          # self.n_samples_valid,
                                          # self.x_resize_valid,
                                          # f.vector_dim)
            # x_train_sequence.append(x_t)
            # x_valid_sequence.append(x_v)
        # x_train_static = []
        # x_valid_static = []
        # for f in self.static_feature_set:
            # if isinstance(f, RapVectorFeature):
                # data = [self.train_rap_vecs, self.valid_rap_vecs]
            # else:
                # raise ValueError("Unkown feature type: {}".format(f.__class__))
            # unordered_t, unordered_v = f.extract(*data)
            # x_t, _ = self._reorder_data(unordered_t,
                                        # self.n_batches_train,
                                        # self.n_samples_train,
                                        # self.x_resize_train,
                                        # f.vector_dim)
            # x_v, _ = self._reorder_data(unordered_v,
                                        # self.n_batches_valid,
                                        # self.n_samples_valid,
                                        # self.x_resize_valid,
                                        # f.vector_dim)
            # x_train_static.append(x_t)
            # x_valid_static.append(x_v)

        # return x_train_sequence, y_t, x_valid_sequence, y_v, x_train_static, x_valid_static

    def load_verses(self, train=True):
        filenames = self.train_filenames
        if not train:
            filenames = self.valid_filenames

        all_seq_features = []
        all_labels = []
        all_context_features = []
        for f in filenames:
            for verse in self._load_file_verses(f):
                verse_word_symbols,\
                verse_char_symbols,\
                verse_phones,\
                verse_stresses,\
                verse_rap_vecs = verse
                verse_seq_features = {
                        "chars": verse_char_symbols[:-1],
                        "phones": verse_phones[:-1],
                        "stresses": verse_stresses[:-1]}
                verse_labels = verse_word_symbols[1:]
                verse_context_features = {"rapper0": verse_rap_vecs[0]}
                for r in xrange(1,self.max_nrps):
                    if len(verse_rap_vecs) > r:
                        verse_context_features["rapper{}".format(r)] = verse_rap_vecs[r]
                all_seq_features.append(verse_seq_features)
                all_labels.append(verse_labels)
                all_context_features.append(verse_context_features)
                break
            break
        self.standardize_vector_lengths(all_seq_features)
        return all_seq_features, all_labels, all_context_features

    def standardize_vector_lengths(self, seq_features):
        word_size = self.max_phones_per_word * self.max_prons_per_word
        phone_sz = self.max_phones_per_word

        for verse in seq_features:
            chars = verse["chars"]
            standard_chars = -1 * np.ones((len(chars), self.max_word_len), dtype=np.int32)
            for i, s in enumerate(chars):
                try:
                    standard_chars[i, :s.shape[0]] = s
                except:
                    pdb.set_trace()
            verse["chars"] = standard_chars

            for feat in ['phones', 'stresses']:
                phone_data = verse[feat]
                vectors = -1 * np.ones((len(phone_data), word_size), dtype=np.int32)
                for i, s in enumerate(phone_data):
                    for j, pron in enumerate(s):
                        vectors[i, j * phone_sz: j * phone_sz + len(pron)] = pron
                verse[feat] = vectors

    def make_verse_instance(self, seq_features, labels, context_features):
        # for n-dimensional stuff
        # flatten and encode original dimensions as context
        example = tf.train.SequenceExample()
        for key, vector in context_features.iteritems():
            for v in vector.astype(np.int64):
                example.context.feature[key].int64_list.value.append(v)

        for key, vector in seq_features.iteritems():
            for v in vector.shape:
                example.context.feature[key+".shape"].int64_list.value.append(v)
            flat = vector.flatten().astype(np.int64)
            feat = example.feature_lists.feature_list[key]
            for v in flat:
                feat.feature.add().int64_list.value.append(v)

        labelfeat = example.feature_lists.feature_list['labels']
        for v in labels:
            labelfeat.feature.add().int64_list.value.append(v)
        return example

    def save_tf_record_files(self, train_file="data/tf_train_data.txt",
                             valid_file="data/tf_valid_data.txt",
                             test_file="data/tf_test_data.txt"):
        for f in [train_file, valid_file]:
            writer = tf.python_io.TFRecordWriter(f)
            all_seq_features, all_labels, all_context_features =\
                    self.load_verses(train=f==train_file)
            for verse, verse_seq_feats in enumerate(all_seq_features):
                verse_labels = all_labels[verse]
                verse_context_features = all_context_features[verse]
                verse_ex = self.make_verse_instance(verse_seq_feats, verse_labels, verse_context_features)
                writer.write(verse_ex.SerializeToString())
            writer.close()

        # don't use test file yet


    def read_and_decode_single_example(self, filename="data/tf_train_data.txt"):

        filename_queue = tf.train.string_input_producer([filename],
                                                num_epochs=None)

        reader = tf.TFRecordReader()
        _, ex = reader.read(filename_queue)

        context_features = {
            "rapper0": tf.FixedLenFeature([self.len_rapper_vector], dtype=tf.int64),
            "phones.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "stresses.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "chars.shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }

        sequence_features = {
            "phones": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "stresses": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return context_parsed, sequence_parsed


    def _split_phone_stress(self, phone_stress):
        try:
            stress = int(phone_stress[-1])
        except:
            return phone_stress, 0
        else:
            return phone_stress[:-1], stress

    def _extract_phones(self, word):
        prons = self._find_prons(word)
        phone_pronunciations = []
        stress_pronunciations = []
        for phone_stresses in prons:
            word_phones = np.zeros(len(phone_stresses))
            word_stresses = np.zeros(len(phone_stresses))
            for i, phone_stress in enumerate(phone_stresses):
                phone, stress = self._split_phone_stress(phone_stress)
                try:
                    word_phones[i] = self.phone2int[phone]
                except:
                    pdb.set_trace()
                word_stresses[i] = stress
            len_phones = len(word_phones)
            if len_phones > self.max_phones_per_word:
                self.max_phones_per_word = len_phones
            phone_pronunciations.append(word_phones)
            stress_pronunciations.append(word_stresses)

        len_pron = len(phone_pronunciations)
        if len_pron > self.max_prons_per_word:
            self.max_prons_per_word = len_pron
        return phone_pronunciations, stress_pronunciations

    def _find_prons(self, word):
        # if len(word) == 1:
            # pdb.set_trace()
        phones_ = pronouncing.phones_for_word
        # TODO: check on this
        # numbers and punc formatting should already have happened

        # num = r'\d+(?:,\d+)?'
        # numbers = re.findall(num, word)
        # if numbers:
            # for num in numbers:
                # word = word.replace(num, ' {} '.format(self._word_numbers(num)))
        # punc_split = re.split('\W+', word)
        full_prons = []
        #for i, subword in enumerate(punc_split):
        lowered = word.lower()
        norm_prons = [p.split() for p in phones_(lowered)]
        if word.isupper():
            acro_pron = []
            for c in lowered:
                if c == 'a':
                    pron = phones_(c)[1]
                    acro_pron.extend(pron.split())
                elif c:
                    pron = phones_(c)[0]
                    acro_pron.extend(pron.split())
            all_subprons = [acro_pron] + norm_prons
        else:
            all_subprons = norm_prons
        if i == 0:
            full_prons = self._remove_dups(all_subprons)
        elif len(all_subprons) > 0:
            new_full_prons = []
            for subpron in all_subprons:
                for pron in full_prons:
                    try:
                        new_full_prons.append(pron + subpron)
                    except:
                        pdb.set_trace()
            full_prons = self._remove_dups(new_full_prons)

        return full_prons

    def _remove_dups(self, l):
        return [s for i, s in enumerate(l)
                if s in l[:i]]

    def _word_numbers(self, num):
        num = num.replace(',','')
        return self.inflect_num_to_words(num)


if __name__ == '__main__':
    # filenames = ['data/test_ints.txt']
    filenames = ['data/test_rap_with_nrp.txt']
    filenames = all_filenames("data/lyric_files/Tyler, The Creator")
    train_ratio = .8
    train_indices = np.random.choice(np.arange(len(filenames)),
                                     replace=False,
                                     size=int(len(filenames)*train_ratio))
    train_filenames = [f for i, f in enumerate(filenames) if i in train_indices]
    valid_filenames = [f for i, f in enumerate(filenames) if i not in train_indices]
    # test_rap = range(100)
    # test_rap = ' '.join([str(x) for x in test_rap])
    # with open('data/test_ints.txt', 'wb') as f:
        # f.write(test_rap)

    extractor = RapFeatureExtractor(train_filenames=train_filenames[0:1],
                                    valid_filenames=valid_filenames[0:1],
                                    batch_size=50,
                                    model_word_len=50,
                                    gzipped=False)
    #extractor.save_tf_record_files()
    context_parsed, sequence_parsed = extractor.read_and_decode_single_example()
    print context_parsed, sequence_parsed
    #sequence = tf.contrib.learn.run_n([context_parsed, sequence_parsed], n=1, feed_dict=None)
    sess = tf.Session()

    # Required. See below for explanation
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # grab examples back.
    # first example from file
    context, sequence = sess.run([context_parsed, sequence_parsed])
    phones_shape = context['phones.shape']
    phones = sequence['phones']
    reshaped = phones.reshape(phones_shape)
    pdb.set_trace()
    context2, sequence2 = sess.run([context_parsed, sequence_parsed])
