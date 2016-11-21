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
import string
from pymongo import MongoClient
client = MongoClient()

# TODO: just use first pronunciation for words
# TODO: see if there's an easy way to load in unstandardized vectors from verse SequenceExamples, and have tensor flow do batching
# TODO: don't bother standardizing word/phone lengths, let tensorflow take care of it at a batch level
# TODO: make sure words are formatted same way as in find_pronunciations and generate_lyric_files

class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
    phone2int = {p: i for i, p in enumerate(all_phones)}

    def __init__(self,
                 train_filenames=None,
                 valid_filenames=None,
                 max_rappers_per_verse=3,
                 sequence_feature_set=None,
                 static_feature_set=None,
                 gzipped=False,
                 from_config=False,
                 config_file='data/config_train.p'):
        self.db = client['lil-neuron-db']
        self.train_filenames = train_filenames
        self.valid_filenames = valid_filenames
        self.gzipped = gzipped

        self.rapper_vectors = self._load_rap_vecs()

        self.char2sym = {}
        self.sym2char = {}
        self.special_symbols = {}
        self.max_word_len = 1
        self.max_prons_per_word = 1
        self.max_phones_per_word = 1
        self.max_nrps = max_rappers_per_verse

        self.word_to_intlist = {}
        self.word_to_int = {}
        self.int_to_word = {}
        self.word_int_to_pron = {}
        self.config_file = config_file
        self._load_special_symbols(from_config)
        self._load_word_dictionaries()

        # inflect_engine = inflect.engine()
        # self.inflect_num_to_words = inflect_engine.number_to_words



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

        # if sequence_feature_set:
            # self.sequence_feature_set = sequence_feature_set
        # else:
            # self.sequence_feature_set = [RawWordsAsChars(),
                                         # RawStresses(),
                                         # RawPhonemes()]
        # if static_feature_set:
            # self.static_feature_set = static_feature_set
        # else:
            # self.static_feature_set = [RapVectorFeature()]

    def _load_rap_vecs(self):
        records = self.db.artists.find({'vector':{'$exists':True}}, {'vector':True, 'name':True})
        rap_vecs = {}
        for r in records:
            rap_vecs[r['name']] = np.array(r['vector'])
        self.len_rapper_vector = len(r['vector'])
        return rap_vecs

    def _load_word_dictionaries(self):
        records = self.db.word_to_dwordint.find()
        for r in records:
            self.word_to_intlist[r['word']] = r['int_list']

        records = self.db.dword_to_int.find()
        startsym = self.vocab_length
        for r in records:
            sym = startsym + r['int']
            self.word_to_int[r['word']] = sym
            self.int_to_word[sym] = r['word']
            if len(r['prons']):
                self.word_int_to_pron[sym] = r['prons'][0].split()
            else:
                self.word_int_to_pron[sym] = []

        records = self.db.slang_words.find()
        for r in records:
            sym = startsym + r['sym']
            self.word_to_intlist[r['word']] = sym
            self.word_to_int[r['word']] = sym
            self.int_to_word[sym] = r['word']
            if 'pronunciation' in r:
                self.word_int_to_pron[sym] = r['pronunciation']
            else:
                self.word_int_to_pron[sym] = []
        self.unknown_prons = set(self.int_to_word[w] for w,p in self.word_int_to_pron.iteritems() if p == '')

    def _load_special_symbols(self, from_config=False):
        self.special_symbols = {
        }
        if from_config:
            self._load_syms_from_config()
        special_symbols = ['<eos>', '<eov>', '<nrp>']
        for s in special_symbols:
            sym = self.vocab_length + 1
            self.word_to_int[s] = sym
            self.int_to_word[sym] = s

            if not from_config:
                self.char2sym[s] = sym
                self.sym2char[sym] = s
                self.special_symbols[s] = sym

    # TODO: save syms to mongo and load from there
    def _load_syms_from_config(self):
        with open(self.config_file, 'rb') as f:
            data = pickle.load(f)
            self.sym2char = data['sym2char']
            self.char2sym = data['char2sym']
            self.special_symbols = data['special_symbols']

    def _save_syms_to_config(self):
        with open(self.config_file, 'wb') as f:
            data = {}
            data['sym2char'] = self.sym2char
            data['char2sym'] = self.char2sym
            data['special_symbols'] = self.special_symbols
            pickle.dump(data, f)

    @property
    def vocab_length(self):
        if len(self.int_to_word):
            return max(self.int_to_word.keys())
        else:
            return 0

    @property
    def char_vocab_length(self):
        return len(self.sym2char)

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
            return [word_feat, char_feat, char_feat, char_feat, rapper_vectors, False]

        words = line.split()

        if len(words) == 1 and words[0] == '':
            word_feat = [self.special_symbols['<eov>']]
            char_feat = [np.array(word_feat)]
            return [word_feat, char_feat, char_feat, char_feat, None, True]

        if len(words) == 1 and words[0] in self.special_symbols:
            word_feat = [self.special_symbols[words[0]]]
            char_feat = [np.array(word_feat)]
            eov = words[0] == '<eov>'
            return [word_feat, char_feat, char_feat, char_feat, None, eov]

        line_word_syms = []
        line_char_syms = []
        line_phones = []
        line_stresses = []
        for i, word in enumerate(words):
            dictwords, syms = self._get_dict_words(word)
            for i, dword in enumerate(dictwords):
                cur_sym = syms[i]
                line_word_syms.append(cur_sym)
                word_symbols = np.zeros(len(dword), dtype=np.int32)
                word_phones, word_stresses = self._extract_phones(cur_sym)

                for j, char in enumerate(dword):
                    if char not in self.char2sym:
                        cur_sym = self.char_vocab_length + 1
                        self.char2sym[char] = cur_sym
                        self.sym2char[cur_sym] = char
                        word_symbols[j] = cur_sym
                    else:
                        word_symbols[j] = self.char2sym[char]
                line_char_syms.append(word_symbols)
                line_phones.append(word_phones)
                line_stresses.append(word_stresses)

        line_word_syms.append(self.special_symbols['<eos>'])
        line_char_syms.append(np.array([self.special_symbols['<eos>']]))
        line_phones.append(np.array([self.special_symbols['<eos>']]))
        line_stresses.append(np.array([self.special_symbols['<eos>']]))
        return line_word_syms, line_char_syms, line_phones, line_stresses, None, False

    def _get_dict_words(self, word):

        trans = [' ']*len(string.punctuation)
        trans[6] = "'"
        trans[11] = ","
        trans[13] = "."
        trans = ''.join(trans)
        replace_punctuation = string.maketrans(string.punctuation, trans)

        translation = word.replace('\xe2\x80\x99',"'")\
                      .replace('\xd1\x81', 'c')\
                      .replace('\xe2\x80\xa6', '')\
                      .lower()\
                      .translate(replace_punctuation)\
                      .split()
        dwords = []
        syms = []
        for w in translation:
            try:
                int_dwords = self.word_to_intlist[w]
            except:
                pdb.set_trace()
                syms.append(1e7)
                dwords.append('<unk>')
                continue
            if not isinstance(int_dwords, list):
                int_dwords = [int_dwords]
            syms.extend(int_dwords)
            [dwords.append(self.int_to_word[i]) for i in int_dwords]
        return dwords, syms

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
                verse_stresses.extend(stresses)
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

    def load_verses(self, train=True):
        filenames = self.train_filenames
        if not train:
            filenames = self.valid_filenames

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
                yield verse_seq_features, verse_labels, verse_context_features
        #self.standardize_vector_lengths(all_seq_features)
        # return all_seq_features, all_labels, all_context_features

    # def standardize_vector_lengths(self, seq_features):
        # word_size = self.max_phones_per_word * self.max_prons_per_word
        # phone_sz = self.max_phones_per_word

        # for verse in seq_features:
            # chars = verse["chars"]
            # standard_chars = -1 * np.ones((len(chars), self.max_word_len), dtype=np.int32)
            # for i, s in enumerate(chars):
                # standard_chars[i, :s.shape[0]] = s
            # verse["chars"] = standard_chars

            # for feat in ['phones', 'stresses']:
                # phone_data = verse[feat]
                # vectors = -1 * np.ones((len(phone_data), self.max_phones_per_word), dtype=np.int32)
                # for i, s in enumerate(phone_data):
                    # vectors[i, :s.shape[0]] = s
                # verse[feat] = vectors

    def make_verse_instance(self, seq_features, labels, context_features):
        # for n-dimensional stuff
        # flatten and encode original dimensions as context
        example = tf.train.SequenceExample()
        for key, vector in context_features.iteritems():
            for v in vector.astype(np.int64):
                example.context.feature[key].int64_list.value.append(v)

        verse_length = len(seq_features.values()[0])
        example.context.feature['verse_length'].int64_list.value.append(verse_length)

        for key, word_vectors in seq_features.iteritems():
            assert len(word_vectors) == verse_length
            word_vector_lengths =[len(v) for v in word_vectors]
            longest = max(word_vector_lengths)
            shape = (len(word_vectors), longest)
            for v in shape:
                example.context.feature[key+".shape"].int64_list.value.append(v)
            feat = example.feature_lists.feature_list[key]
            length_feat = example.feature_lists.feature_list[key+".lengths"]
            for i, v in enumerate(word_vectors):
                length_feat.feature.add().int64_list.value.append(word_vector_lengths[i])
                for value in v:
                    feat.feature.add().int64_list.value.append(int(value))
                for value in xrange(len(v), longest):
                    feat.feature.add().int64_list.value.append(0)

        word_length = longest
        example.context.feature['word_length'].int64_list.value.append(word_length)

        labelfeat = example.feature_lists.feature_list['labels']
        for v in labels:
            labelfeat.feature.add().int64_list.value.append(int(v))
        return example

    def save_tf_record_files(self, train_file="data/tf_train_data.txt",
                             valid_file="data/tf_valid_data.txt",
                             test_file="data/tf_test_data.txt"):
        for f in [train_file, valid_file]:
            writer = tf.python_io.TFRecordWriter(f)
            #all_seq_features, all_labels, all_context_features =\
            for seq, label, context in self.load_verses(train=f==train_file):
                verse_ex = self.make_verse_instance(seq, label, context)
                writer.write(verse_ex.SerializeToString())
            writer.close()
        self._save_syms_to_config()

        # don't use test file yet


    def read_and_decode_single_example(self, filename="data/tf_train_data.txt", num_epochs=None):

        filename_queue = tf.train.string_input_producer([filename],
                                                num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, ex = reader.read(filename_queue)

        # TODO: other rappers
        context_features = {
            "rapper0": tf.FixedLenFeature([self.len_rapper_vector], dtype=tf.int64),
            "phones.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "stresses.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "chars.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "verse_length": tf.FixedLenFeature([1], dtype=tf.int64),
            "word_length": tf.FixedLenFeature([1], dtype=tf.int64)
        }

        sequence_features = {
            "phones": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "stresses": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),

            "phones.lengths": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "stresses.lengths": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars.lengths": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        init_op_local = tf.initialize_local_variables()

        return context_parsed, sequence_parsed, init_op_local


    def _split_phone_stress(self, phone_stress):
        try:
            stress = int(phone_stress[-1])
        except:
            return phone_stress, 1
        else:
            if stress > 2:
                pdb.set_trace()
            return phone_stress[:-1], stress + 1

    def _extract_phones(self, wordint):
        phone_stresses = self._find_prons(wordint)

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

        return word_phones, word_stresses

    def _find_prons(self, wordint):
        # word_int_to_prons should contain both slang and non-slang
        return self.word_int_to_pron[wordint]

        # phones_ = pronouncing.phones_for_word
        # # TODO: check on this
        # # numbers and punc formatting should already have happened

        # # num = r'\d+(?:,\d+)?'
        # # numbers = re.findall(num, word)
        # # if numbers:
            # # for num in numbers:
                # # word = word.replace(num, ' {} '.format(self._word_numbers(num)))
        # # punc_split = re.split('\W+', word)
        # full_prons = []
        # #for i, subword in enumerate(punc_split):
        # lowered = word.lower()
        # norm_prons = [p.split() for p in phones_(lowered)]
        # if word.isupper():
            # acro_pron = []
            # for c in lowered:
                # if c == 'a':
                    # pron = phones_(c)[1]
                    # acro_pron.extend(pron.split())
                # elif c:
                    # pron = phones_(c)[0]
                    # acro_pron.extend(pron.split())
            # all_subprons = [acro_pron] + norm_prons
        # else:
            # all_subprons = norm_prons
        # if i == 0:
            # full_prons = self._remove_dups(all_subprons)
        # elif len(all_subprons) > 0:
            # new_full_prons = []
            # for subpron in all_subprons:
                # for pron in full_prons:
                    # try:
                        # new_full_prons.append(pron + subpron)
                    # except:
                        # pdb.set_trace()
            # full_prons = self._remove_dups(new_full_prons)

        # return full_prons

    # def _remove_dups(self, l):
        # return [s for i, s in enumerate(l)
                # if s in l[:i]]

    # def _word_numbers(self, num):
        # num = num.replace(',','')
        # return self.inflect_num_to_words(num)


if __name__ == '__main__':
    # filenames = ['data/test_ints.txt']
    filenames = ['data/test_rap_with_nrp.txt']
    filenames = all_filenames("data/lyric_files/Tyler, The Creator")
    #filenames = all_filenames("data/lyric_files")
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

    print len(train_filenames)
    print len(valid_filenames)
    extractor = RapFeatureExtractor(train_filenames=train_filenames[0:1],
                                    valid_filenames=valid_filenames[0:1],
                                    from_config=False,
                                    config_file='data/config_test.p')
    extractor.save_tf_record_files(train_file="data/tf_train_data_test.txt",
                                   valid_file="data/tf_valid_data_test.txt")
