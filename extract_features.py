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
import gzip
from features import PhonemeFeature, WordFeature, StressFeature
from features import RawStresses, RawPhonemes, RawWordsAsChars

EOV = -2
EOP = -3
EOS = -4
# NRP = -5
# TODO: Also include EOV (end of verse), NRP (new rapper),
# NRP is followed by an integer (which becomes an embedding)
# representing the rapper who is rapping the current verse

# TODO: all these are simple loops and should be in Cython


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
                 word2sym={},
                 sym2word={},
                 char2sym={},
                 sym2char={},
                 char_vocab_length=0,
                 vocab_length=0,
                 feature_set=None,
                 gzipped=True,
                 pickle_file=None):
        self.train_filenames = train_filenames
        self.valid_filenames = valid_filenames
        self.pck_data_file = pickle_file
        if self.pck_data_file is None:
            self.pck_data_file = os.path.join("data", "formatted_data.p")
        self.gzipped = gzipped

        self.batch_size = batch_size
        self.model_word_len = model_word_len
        self.char2sym = char2sym
        self.sym2char = sym2char
        self.word2sym = word2sym
        self.sym2word = sym2word
        self.char_vocab_length = char_vocab_length
        self.vocab_length = vocab_length
        self.max_word_len = 0
        self.max_prons_per_word = 0
        self.max_phones_per_word = 0

        self.special_symbols = {
        }
        special_symbols = ['<eos>', '<eov>', '<nrp>']
        for s in special_symbols:
            self.word2sym[s] = self.vocab_length
            self.sym2word[self.vocab_length] = s
            self.special_symbols[s] = np.array(self.vocab_length)
            self.vocab_length += 1

        self.train_word_syms = []
        self.train_char_syms = []
        self.train_phone_syms = []
        self.train_stress_syms = []
        self.valid_word_syms = []
        self.valid_char_syms = []
        self.valid_phone_syms = []
        self.valid_stress_syms = []

        if feature_set:
            self.feature_set = feature_set
        else:
            self.feature_set = [RawWordsAsChars(),
                                RawStresses(),
                                RawPhonemes()]


    def _process_line(self, line):
        line = line.lstrip().rstrip()
        words = line.split(" ")
        # TODO: Parse NRP
        if len(words) == 1 and words[0] == '':
            return ([self.special_symbols['<eov>']],
                    [np.array([self.special_symbols['<eov>']])],
                    [[np.array([self.special_symbols['<eov>']])]],
                    [[np.array([self.special_symbols['<eov>']])]])
        if len(words) == 1 and words[0] in self.special_symbols:
            return ([self.special_symbols(words[0])],
                    [np.array([self.special_symbols(words[0])])],
                    [[np.array([self.special_symbols(words[0])])]],
                    [[np.array([self.special_symbols(words[0])])]])
        line_word_syms = []
        line_char_syms = []
        line_phones = []
        line_stresses = []
        for i, word in enumerate(words):
            if word not in self.word2sym:
                cur_word = self.vocab_length
                self.word2sym[word] = cur_word
                self.sym2word[cur_word] = word
                self.vocab_length += 1
            line_word_syms.append(self.word2sym[word])
            word_symbols = np.zeros(len(word), dtype=np.int32)
            word_phones, word_stresses = self._extract_phones(word)
            if len(word) > self.max_word_len:
                self.max_word_len = len(word)
            for j, char in enumerate(word):
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
        return line_word_syms, line_char_syms, line_phones, line_stresses

    def _load_file(self, fname, valid=False):
        if self.gzipped:
            open_func = gzip.open
        else:
            open_func = open
        all_word_symbols = []
        all_char_symbols = []
        all_phones = []
        all_stresses = []
        with open_func(fname, 'rb') as f:
            for line in f:
                word_symbols, char_symbols, phones, stresses = self._process_line(line)
                all_word_symbols.extend(word_symbols)
                all_char_symbols.extend(char_symbols)
                all_phones.extend(phones)
                all_stresses.extend(stresses)
        if valid:
            self.valid_word_syms.extend(all_word_symbols)
            self.valid_char_syms.extend(all_char_symbols)
            self.valid_phone_syms.extend(all_phones)
            self.valid_stress_syms.extend(all_stresses)
        else:
            self.train_word_syms.extend(all_word_symbols)
            self.train_char_syms.extend(all_char_symbols)
            self.train_phone_syms.extend(all_phones)
            self.train_stress_syms.extend(all_stresses)

    def _save_data_to_pickle(self):
        data = [self.train_word_syms, self.valid_word_syms,
                self.train_char_syms, self.valid_char_syms,
                self.train_phone_syms, self.valid_phone_syms,
                self.train_stress_syms, self.valid_stress_syms]
        metadata = [self.char_vocab_length, self.char2sym, self.sym2char,
                    self.vocab_length, self.word2sym, self.sym2word]
        with open(self.pck_data_file, 'wb') as f:
            pickle.dump([data, metadata], f)

    def _load_data_from_pickle(self):
        if os.path.isfile(self.pck_data_file):
            with open(self.pck_data_file, 'rb') as f:
                return pickle.load(f)
        return None

    def _load_data(self):
        data = self._load_data_from_pickle()
        if data is None or len(data) != 2:
            for f in self.train_filenames:
                self._load_file(f)
            for f in self.train_filenames:
                self._load_file(f, valid=True)
            self._standardize_word_lengths()
            self._save_data_to_pickle()
        else:
            data, metadata = data
            self.char_vocab_length, self.char2sym, self.sym2char = metadata[:3]
            self.vocab_length, self.word2sym, self.sym2word = metadata[3:]
            self.train_word_syms, self.valid_word_syms = data[:2]
            self.train_char_syms, self.valid_char_syms = data[2:4]
            self.train_phone_syms, self.valid_phone_syms = data[4:6]
            self.train_stress_syms, self.valid_stress_syms = data[6:]
        self.x_resize_train, self.n_samples_train, self.n_batches_train =\
            self._calc_num_batches(len(self.train_char_syms))
        self.x_resize_valid, self.n_samples_valid, self.n_batches_valid =\
            self._calc_num_batches(len(self.valid_char_syms))

    def _calc_num_batches(self, len_symbols):
        div = len_symbols // (self.batch_size * self.model_word_len)
        x_resize = div * self.model_word_len * self.batch_size
        n_samples = x_resize // self.model_word_len
        n_batches = n_samples // self.batch_size
        return x_resize, n_samples, n_batches

    def _standardize_word_lengths(self):
        self.train_word_syms = np.array(self.train_word_syms)
        self.valid_word_syms = np.array(self.valid_word_syms)
        new_word_vectors = []
        for word_data in [self.train_char_syms, self.valid_char_syms]:
            vectors = -1 * np.ones((len(word_data), self.max_word_len))
            for i, s in enumerate(word_data):
                vectors[i, :s.shape[0]] = s
            new_word_vectors.append(vectors)
        self.train_char_syms, self.valid_char_syms = new_word_vectors

        word_size = self.max_phones_per_word * self.max_prons_per_word
        phone_sz = self.max_phones_per_word
        new_phone_vectors = []
        for phone_data in [self.train_phone_syms, self.valid_phone_syms,
                           self.train_stress_syms, self.valid_stress_syms]:
            vectors = -1 * np.ones((len(phone_data), word_size))
            for i, s in enumerate(phone_data):
                for j, pron in enumerate(s):
                    vectors[i, j * phone_sz: j * phone_sz + len(pron)] = pron
            new_phone_vectors.append(vectors)
        self.train_phone_syms, self.valid_phone_syms = new_phone_vectors[:2]
        self.train_stress_syms, self.valid_stress_syms = new_phone_vectors[2:]

    def _reorder_data(self, x_in, n_batches, n_samples, x_resize, vector_dim):
        if x_in.shape[0] % (self.batch_size * self.model_word_len) == 0:
            print(" x_in.shape[0] % (batch_size*model_word_len) == 0 -> x_in is "
                  "set to x_in = x_in[:-1]")
            x_in = x_in[:-1]

        if vector_dim > 0:
            args = [n_samples, self.model_word_len, vector_dim]
        else:
            args = [n_samples, self.model_word_len]
        targets = x_in[1: x_resize + 1].reshape(*args)

        x_out = x_in[:x_resize].reshape(*args)

        out = np.zeros(n_samples, dtype=int)

        for i in range(n_batches):
            val = range(i, n_batches * self.batch_size + i, n_batches)
            out[i * self.batch_size: (i + 1) * self.batch_size] = val

        x_out = x_out[out]
        targets = targets[out]

        return x_out.astype('int32'), targets.astype('int32')

    def extract(self):
        self._load_data()
        x_train = []
        x_valid = []
        unordered_raw_t = self.train_word_syms
        _, y_t = self._reorder_data(unordered_raw_t,
                                    self.n_batches_train,
                                    self.n_samples_train,
                                    self.x_resize_train,
                                    0)
        unordered_raw_v = self.valid_word_syms
        _, y_v = self._reorder_data(unordered_raw_v,
                                    self.n_batches_train,
                                    self.n_samples_train,
                                    self.x_resize_train,
                                    0)
        for f in self.feature_set:
            if isinstance(f, PhonemeFeature):
                data = [self.train_phone_syms, self.valid_phone_syms]
            elif isinstance(f, StressFeature):
                data = [self.train_stress_syms, self.valid_stress_syms]
            elif isinstance(f, WordFeature):
                data = [self.train_char_syms, self.valid_char_syms]
            else:
                raise ValueError("Unkown feature type: {}".format(f.__class__))
            unordered_t, unordered_v = f.extract(*data)
            x_t, _ = self._reorder_data(unordered_t,
                                          self.n_batches_train,
                                          self.n_samples_train,
                                          self.x_resize_train,
                                          f.vector_dim)
            x_v, _ = self._reorder_data(unordered_v,
                                          self.n_batches_valid,
                                          self.n_samples_valid,
                                          self.x_resize_valid,
                                          f.vector_dim)
            x_train.append(x_t)
            x_valid.append(x_v)
        return x_train, y_t, x_valid, y_v

    def _split_phone_stress(self, phone_stress):
        try:
            stress = int(phone_stress[-1])
        except:
            return phone_stress, 0
        else:
            return phone_stress[:-1], stress

    def _extract_phones(self, word):
        prons = pronouncing.phones_for_word(word)
        phone_pronunciations = []
        stress_pronunciations = []
        for pr in prons:
            phone_stresses = pr.split(' ')
            word_phones = np.zeros(len(phone_stresses))
            word_stresses = np.zeros(len(phone_stresses))
            for i, phone_stress in enumerate(phone_stresses):
                phone, stress = self._split_phone_stress(phone_stress)
                word_phones[i] = self.phone2int[phone]
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

if __name__ == '__main__':
    # filenames = ['data/test_ints.txt']
    filenames = ['data/test_rap.txt']
    test_rap = range(100)
    test_rap = ' '.join([str(x) for x in test_rap])
    with open('data/test_ints.txt', 'wb') as f:
        f.write(test_rap)

    batch_size = 2
    model_word_len = 3
    word_embedding_size = 1
    char2sym = {}
    sym2char={}
    char_vocab_length = [0]
    extractor = RapFeatureExtractor(train_filenames=filenames,
                                    valid_filenames=filenames,
                                    batch_size=3,
                                    model_word_len=3,
                                    char2sym=char2sym,
                                    sym2char=sym2char,
                                    gzipped=False)
    x_train, y_train, x_valid, y_valid = extractor.extract()
    # print x_train, y_train
