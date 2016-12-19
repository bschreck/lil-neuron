"""
Pronunciation features:
    - phonemes as stresses
    - raw phonemes, no stresses
    - raw phonemes, broken up into groups separated by stresses
    - raw phonemes, broken up into groups separated by words
    - raw phonemes, broken up into groups separated by phrases or lines
    - rhyme scheme as categorical variables (mark each rhyme with same category)
"""
import numpy as np
import cPickle as pickle
import pdb
import sys
import tensorflow as tf
import string
import re
try:
    from pymongo import MongoClient
    client = MongoClient()
except:
    pass


class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])

    def __init__(self,
                 train_corpus=None,
                 valid_corpus=None,
                 test_corpus=None,
                 max_rappers_per_verse=3,
                 sequence_feature_set=None,
                 static_feature_set=None,
                 from_config=False,
                 spellcheck_dicts='data/word_dicts.p',
                 config_file='data/config_train.p'):
        if not from_config:
            self.db = client['lil-neuron-db']
        self.train_corpus = train_corpus
        self.valid_corpus = valid_corpus
        self.test_corpus = test_corpus
        self.max_nrps = max_rappers_per_verse
        self.config_file = config_file

        self.phone2sym = {}
        self.sym2phone = {}
        self.word_to_int = {}
        self.int_to_word = {}
        self.glove_word_to_int = {}

        self._load_spellcheck_dict(spellcheck_dicts)

        if from_config:
            self._load_syms_from_config()
        else:
            self._load_special_symbols()
            self._load_rap_vecs()
            self._load_phones()
        self.len_rapper_vector = len(self.rapper_vectors.values()[0])

    def _load_spellcheck_dict(self, filename):
        with open(filename, 'rb') as f:
            dicts = pickle.load(f)
            self.word_to_dword = dicts['word_to_dword']

    def _load_rap_vecs(self):
        records = self.db.artists.find({'vector': {'$exists': True}},
                                       {'vector': True, 'name': True})
        rap_vecs = {}
        for r in records:
            name = r['name']
            rap_vecs[name] = np.array(r['vector'])
            self.get_word_sym(name)
        self.rapper_vectors = rap_vecs

    def _load_phones(self):
        for p in self.all_phones:
            self.get_phone_sym(p)


    def _load_special_symbols(self):
        special_symbols = ['<eos>', '<eov>', '<nrp>', '<eor>', '<unk>']
        for s in special_symbols:
            self.get_word_sym(s)

    def get_sym(self, full, full2sym, vocab_length, sym2full=None):
        if full in full2sym:
            return full2sym[full]
        else:
            sym = vocab_length + 1
            full2sym[full] = sym
            if sym2full is not None:
                sym2full[sym] = full
            return sym

    def get_word_sym(self, word):
        return self.get_sym(word, self.word_to_int, self.vocab_length, sym2full=self.int_to_word)

    def get_phone_sym(self, phone):
        return self.get_sym(phone, self.phone2sym, self.phone_vocab_length, sym2full=self.sym2phone)

    # TODO: save syms to mongo and load from there
    def _load_syms_from_config(self):
        with open(self.config_file, 'rb') as f:
            data = pickle.load(f)
            self.sym2phone = data['sym2phone']
            self.phone2sym = data['phone2sym']
            self.word_to_int     =   data['word_to_int']
            self.int_to_word     =   data['int_to_word']
            self.glove_word_to_int = data['glove_word_to_int']
            self.rapper_vectors =   data['rapper_vectors']

    def _save_syms_to_config(self):
        with open(self.config_file, 'wb') as f:
            data = {}
            data['sym2phone']  = self.sym2phone
            data['phone2sym']  = self.phone2sym
            data['word_to_int'] = self.word_to_int
            data['int_to_word'] = self.int_to_word
            data['glove_word_to_int'] = self.gen_glove_dict(self.word_to_int)
            data['rapper_vectors'] = self.rapper_vectors
            pickle.dump(data, f)

    def gen_glove_dict(self, word_dict):
        glove_dict = {}
        for w, i in word_dict.iteritems():
            number_convert = re.sub('\d+', '#', w)
            glove_dict[number_convert] = i
        return glove_dict

    @property
    def vocab_length(self):
        return len(self.word_to_int)

    @property
    def phone_vocab_length(self):
        return len(self.phone2sym)

    def _add_rappers(self, *rappers):
        # TODO: how to deal with rapper words if no vectors associated with them?, (0-vectors?)

        # TODO: also include context feature for pronunciation
        # of rapper, and characters in the word
        rapper_vectors = []
        for rapper in rappers:
            if rapper in self.rapper_vectors:
                rapper_vector = self.rapper_vectors[rapper]
                rapper_vectors.append(rapper_vector)
            else:
                rapper_vectors.append(np.zeros(self.len_rapper_vector))
        if len(rapper_vectors) > self.max_nrps:
            rapper_vectors = rapper_vectors[:self.max_nrps]
        return rapper_vectors

    def _extract_nrps(self, word):
        if word.startswith('<nrp:'):
            words = word.split('<nrp:')[1][:-1]
            rappers = words.split(';')
            return self._add_rappers(*rappers)
        return None

    def get_word_from_int(self, sym):
        return self.int_to_word[sym]

    def _load_file_verses(self, fname, valid=False):
        verse_word_symbols = []
        verse_rap_vecs = None
        with open(fname, 'rb') as f:
            words = f.read().split()
            for word in words:
                rap_vecs = self._extract_nrps(word)
                if rap_vecs is not None and verse_rap_vecs is None:
                    verse_rap_vecs = rap_vecs

                # check spellcheck dict
                if word in self.word_to_dword:
                    word = self.word_to_dword[word]

                word_symbol = self.get_word_sym(word)
                verse_word_symbols.append(word_symbol)

                if word == '<eov>':
                    seq, labels, context = self.make_feature_dict(verse_word_symbols,
                                                                  verse_rap_vecs,
                                                                  as_lm=True)
                    yield seq, labels, context
                    verse_word_symbols = []
                if word == '<eor>':
                    verse_rap_vecs = None

    def make_feature_dict(self, words, rap_vecs, as_lm=True):
        syms = words
        labels = words
        if as_lm:
            syms = words[:-1]
            labels = words[1:]
        features = {
            "words": syms,
        }
        assert len(rap_vecs), "Attempt to make a feature without any rappers"
        context = {"rapper0": rap_vecs[0]}
        for r in xrange(1, self.max_nrps):
            if len(rap_vecs) > r:
                context["rapper{}".format(r)] = rap_vecs[r]
            else:
                context["rapper{}".format(r)] = np.zeros(self.len_rapper_vector)
        return features, labels, context

    def gen_features_from_starter(self, rappers, starter):
        rap_vecs = [self.rapper_vectors[r] for r in rappers]
        words = [self.get_word_sym(w) for w in starter]

        features, labels, context = self.make_feature_dict(words,
                                                           rap_vecs,
                                                           as_lm=False)
        actual_input_data = {"labels": np.array(labels)[np.newaxis, :],
                             "words": np.array(labels)[np.newaxis, :]}
        verse_length = np.array([[len(features.values()[0])]])
        actual_input_data["verse_length"] = verse_length


        for key, array in context.iteritems():
            expanded = np.tile(array[np.newaxis, :], [verse_length[0,0], 1])
            actual_input_data[key] = expanded[np.newaxis, :]

        verse_ex = self.make_verse_instance(features, labels, context)
        tensor_dict, init_op_local = self.read_and_decode_single_example(verse_length, from_example=verse_ex)
        # Add a batch of 1
        for key, value in tensor_dict.iteritems():
            tensor_dict[key] = tf.expand_dims(value, 0)
        return tensor_dict, actual_input_data, init_op_local


    def tensorize(self, data, add_batch=True, name=None):
        if add_batch:
            data = data[np.newaxis, :]
        return tf.convert_to_tensor(data, name=name, dtype=tf.int32)

    def update_features(self, sym):
        input_data = {}
        input_data["verse_length"] = np.array([[1]])
        input_data["labels"] = np.array([[sym]])
        input_data["words"] = np.array([[sym]])
        return input_data

    def gen_context_from_nrp(self, prev_context, nrp_sym, position):
        if nrp_sym in self.sym2char:
            rapper = self.sym2char[nrp_sym]
            rap_vec = self.rapper_vectors[rapper]
            prev_context["rapper"+str(position)] = tf.cast(tf.tensor(rap_vec), tf.int32)
        return prev_context

    def load_verses(self, ftype):
        if ftype == 'train':
            corpus = self.train_corpus
        elif ftype == 'valid':
            corpus = self.valid_corpus
        else:
            corpus = self.test_corpus

        for verse in self._load_file_verses(corpus):
            yield verse

    def make_verse_instance(self, seq_features, labels, context_features):
        # for n-dimensional stuff
        # flatten and encode original dimensions as context
        example = tf.train.SequenceExample()
        for key, vector in context_features.iteritems():
            for v in vector.astype(np.int64):
                example.context.feature[key].int64_list.value.append(v)

        verse_length = len(seq_features.values()[0])
        assert len(labels) == verse_length
        for v in seq_features.values():
            assert len(v) == verse_length

        example.context.feature['verse_length'].int64_list.value.append(verse_length)

        for key, values in seq_features.iteritems():
            feat = example.feature_lists.feature_list[key]
            for v in values:
                feat.feature.add().int64_list.value.append(int(v))

        labelfeat = example.feature_lists.feature_list['labels']
        for v in labels:
            labelfeat.feature.add().int64_list.value.append(int(v))
        return example

    def save_tf_record_files(self, train_file="data/tf_train_data.txt",
                             valid_file="data/tf_valid_data.txt",
                             test_file="data/tf_test_data.txt"):
        ftype = ["train", "valid", "test"]
        files = [train_file, valid_file, test_file]
        for ftype, f in zip(ftype, files):
            writer = tf.python_io.TFRecordWriter(f)
            for seq, label, context in self.load_verses(ftype):
                verse_ex = self.make_verse_instance(seq, label, context)
                writer.write(verse_ex.SerializeToString())
            writer.close()
        self._save_syms_to_config()

    def read_and_decode_single_example(self, max_num_steps, from_filename=None, from_example=None, num_epochs=None):
        #"data/tf_train_data.txt"
        assert from_filename is not None or from_example is not None
        if from_filename:
            filename_queue = tf.train.string_input_producer([from_filename],
                                                             num_epochs=num_epochs)

            reader = tf.TFRecordReader()
            _, ex = reader.read(filename_queue)
        else:
            ex = from_example.SerializeToString()

        context_features = {
            "verse_length": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        for r in xrange(self.max_nrps):
            context_features["rapper" + str(r)] = tf.FixedLenFeature([self.len_rapper_vector], dtype=tf.int64)

        sequence_features = {
            "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        casted_tensors = self.cast_tensors(context_parsed, sequence_parsed)

        to_batch = {k: v for k, v in casted_tensors.iteritems() if k in sequence_features}

        verse_length = casted_tensors.pop('verse_length')
        context_features = [k for k in casted_tensors if k not in sequence_features]
        for c in context_features:
            multiples = tf.pack([verse_length[0], 1])
            to_batch[c] = tf.tile(tf.expand_dims(casted_tensors[c], 0),
                                  multiples)

        init_op_local = tf.initialize_local_variables()
        return to_batch, init_op_local

    def cast_tensors(self, context, sequence):
        casted = {}
        for key, tensor in context.iteritems():
            casted[key] = tf.cast(tensor, tf.int32)

        for key, tensor in sequence.iteritems():
            ct = tf.cast(tensor, tf.int32)
            casted[key] = ct
        return casted


if __name__ == '__main__':
    train_corpus = 'data/train_corpus.txt'
    valid_corpus = 'data/valid_corpus.txt'
    test_corpus = 'data/test_corpus.txt'
    extractor = RapFeatureExtractor(train_corpus=train_corpus,
                                    valid_corpus=valid_corpus,
                                    test_corpus=test_corpus,
                                    from_config=False,
                                    spellcheck_dicts='data/word_dicts.p',
                                    config_file='data/config_full.p')
    extractor.save_tf_record_files(train_file="data/tf_train_data_full.txt",
                                   valid_file="data/tf_valid_data_full.txt",
                                   test_file="data/tf_test_data_full.txt")
