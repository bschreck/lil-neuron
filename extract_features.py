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
import gzip
import tensorflow as tf
import string
import copy
try:
    from generate_lyric_files import all_filenames
    from pymongo import MongoClient
    client = MongoClient()
except:
    pass

# TODO: just use first pronunciation for words
# TODO: see if there's an easy way to load in unstandardized vectors from verse SequenceExamples, and have tensor flow do batching
# TODO: don't bother standardizing word/phone lengths, let tensorflow take care of it at a batch level
# TODO: make sure words are formatted same way as in find_pronunciations and generate_lyric_files

class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
    all_stresses = [0, 1, 2]

    def __init__(self,
                 train_filenames=None,
                 valid_filenames=None,
                 test_filenames=None,
                 max_rappers_per_verse=3,
                 sequence_feature_set=None,
                 static_feature_set=None,
                 gzipped=False,
                 from_config=False,
                 config_file='data/config_train.p'):
        if not from_config:
            self.db = client['lil-neuron-db']
        self.train_filenames = train_filenames
        self.valid_filenames = valid_filenames
        self.test_filenames = test_filenames
        self.gzipped = gzipped
        self.max_nrps = max_rappers_per_verse
        self.config_file = config_file

        self.char2sym = {}
        self.sym2char = {}
        self.stress2sym = {}
        self.sym2stress = {}
        self.phone2sym = {}
        self.sym2phone = {}
        self.word_to_intlist = {}
        self.word_to_int = {}
        self.int_to_word = {}
        self.word_int_to_pron = {}

        if from_config:
            self._load_syms_from_config()
        else:
            self._load_special_symbols()
            self._load_rap_vecs()
            self._load_phone_stresses()
            self._load_word_dictionaries()
        self.len_rapper_vector = len(self.rapper_vectors.values()[0])
        self.unknown_prons = set(self.int_to_word[w] for w, p in self.word_int_to_pron.iteritems() if len(p) == 0)

    def _load_rap_vecs(self):
        records = self.db.artists.find({'vector': {'$exists': True}},
                                       {'vector': True, 'name': True})
        rap_vecs = {}
        chars_in_rappers = set()
        for r in records:
            name = r['name']
            rap_vecs[name] = np.array(r['vector'])
            chars_in_rappers |= set(name)

            self.get_char_sym(name)
            self.get_phone_sym(name)
            self.get_stress_sym(name)
            self.get_word_sym(name)
        for c in chars_in_rappers:
            self.get_char_sym(c)
        self.rapper_vectors = rap_vecs

    def _load_phone_stresses(self):
        for s in self.all_stresses:
            self.get_stress_sym(s)
        for p in self.all_phones:
            self.get_phone_sym(p)

    def _load_word_dictionaries(self):
        mongo_ints_to_syms = {}
        records = self.db.dword_to_int.find()
        for r in records:
            sym = self.get_word_sym(r['word'])
            mongo_ints_to_syms[r['int']] = sym
            if len(r['prons']):
                self.word_int_to_pron[sym] = r['prons'][0].split()
            else:
                self.word_int_to_pron[sym] = []

        records = self.db.word_to_dwordint.find()
        for r in records:
            self.word_to_intlist[r['word']] = [mongo_ints_to_syms[i] for i in r['int_list']]

        records = self.db.slang_words.find()
        for r in records:
            sym = self.get_word_sym(r['word'])
            if 'pronunciation' in r:
                self.word_int_to_pron[sym] = r['pronunciation']
            else:
                self.word_int_to_pron[sym] = []

    def _load_special_symbols(self, from_config=False):
        self.special_word_symbols = {
        }
        self.special_char_symbols = {
        }
        special_symbols = ['<eos>', '<eov>', '<nrp>', '<eor>', '<unk>']
        for s in special_symbols:
            word_sym = self.get_word_sym(s)
            char_sym = self.get_char_sym(s)
            self.get_phone_sym(s)
            self.get_stress_sym(s)
            self.special_word_symbols[s] = word_sym
            self.word_int_to_pron[word_sym] = []
            self.special_char_symbols[s] = char_sym

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

    def get_char_sym(self, char):
        return self.get_sym(char, self.char2sym, self.char_vocab_length, sym2full=self.sym2char)

    def get_phone_sym(self, phone):
        return self.get_sym(phone, self.phone2sym, self.phone_vocab_length, sym2full=self.sym2phone)

    def get_stress_sym(self, stress):
        return self.get_sym(stress, self.stress2sym, self.stress_vocab_length, sym2full=self.sym2stress)

    # TODO: save syms to mongo and load from there
    def _load_syms_from_config(self):
        with open(self.config_file, 'rb') as f:
            data = pickle.load(f)
            self.sym2char = data['sym2char']
            self.char2sym = data['char2sym']
            self.sym2phone = data['sym2phone']
            self.phone2sym = data['phone2sym']
            self.sym2stress = data['sym2stress']
            self.stress2sym = data['stress2sym']
            self.special_word_symbols = data['special_char_symbols']
            self.special_char_symbols = data['special_word_symbols']
            self.word_to_intlist =   data['word_to_intlist']
            self.word_to_int     =   data['word_to_int']
            self.int_to_word     =   data['int_to_word']
            self.word_int_to_pron =   data['word_int_to_pron']
            self.rapper_vectors =   data['rapper_vectors']

    def _save_syms_to_config(self):
        with open(self.config_file, 'wb') as f:
            data = {}
            data['sym2char'] = self.sym2char
            data['char2sym'] = self.char2sym
            data['sym2phone']  = self.sym2phone
            data['phone2sym']  = self.phone2sym
            data['sym2stress'] = self.sym2stress
            data['stress2sym'] = self.stress2sym
            data['special_word_symbols'] = self.special_word_symbols
            data['special_char_symbols'] = self.special_char_symbols
            data['word_to_intlist'] = self.word_to_intlist
            data['word_to_int'] = self.word_to_int
            data['int_to_word'] = self.int_to_word
            data['word_int_to_pron'] = self.word_int_to_pron
            data['rapper_vectors'] = self.rapper_vectors
            pickle.dump(data, f)

    @property
    def vocab_length(self):
        return len(self.word_to_int)
    @property
    def char_vocab_length(self):
        return len(self.char2sym)
    @property
    def phone_vocab_length(self):
        return len(self.phone2sym)
    @property
    def stress_vocab_length(self):
        return len(self.stress2sym)

    def _add_rappers(self, *rappers):
        # TODO: also include context feature for pronunciation
        # of rapper, and characters in the word
        rapper_char_syms = []
        rapper_word_syms = []
        rapper_vectors = []
        for rapper in rappers:
            if rapper in self.rapper_vectors:
                char_sym = self.char2sym[rapper]
                word_sym = self.word_to_int[rapper]
                rapper_vector = self.rapper_vectors[rapper]
                rapper_vectors.append(rapper_vector)
            else:
                char_sym = self.get_char_sym(rapper)
                word_sym = self.get_char_sym(rapper)
                rapper_vectors.append(np.zeros(self.len_rapper_vector))
            rapper_char_syms.append(char_sym)
            rapper_word_syms.append(word_sym)
        if len(rapper_vectors) > self.max_nrps:
            rapper_word_syms = rapper_word_syms[:self.max_nrps]
            rapper_char_syms = rapper_char_syms[:self.max_nrps]
            rapper_vectors = rapper_vectors[:self.max_nrps]
        return rapper_word_syms, rapper_char_syms, rapper_vectors

    def _nrp_line(self, line):
        if line.startswith('(NRP:'):
            words = line.split('(NRP:')
            rappers = [w.replace(")","").strip() for w in words
                       if w]
            return self._add_rappers(*rappers)
        return None, None, None

    def get_word_from_int(self, sym):
        return self.int_to_word[sym]

    def _process_line(self, line):
        line = line.lstrip().rstrip()
        rapper_word_syms, rapper_char_syms, rapper_vectors = self._nrp_line(line)
        if rapper_word_syms is not None:
            word_feat = [self.special_word_symbols['<nrp>']] + rapper_word_syms + [self.special_word_symbols['<eos>']]
            char_feat = [self.special_char_symbols['<nrp>']] + rapper_char_syms + [self.special_char_symbols['<eos>']]
            char_feat = [np.array([c]) for c in char_feat]
            return [word_feat, char_feat, char_feat, char_feat, rapper_vectors, False]

        words = line.split()

        if len(words) == 1 and words[0] == '':
            word_feat = [self.special_word_symbols['<eov>']]
            char_feat = [np.array([self.special_char_symbols['<eov>']])]
            return [word_feat, char_feat, char_feat, char_feat, None, True]

        if len(words) == 1 and words[0] in self.special_word_symbols:
            word_feat = [self.special_word_symbols[words[0]]]
            char_feat = [np.array([self.special_char_symbols[words[0]]])]
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
                    word_symbols[j] = self.get_char_sym(char)

                line_char_syms.append(word_symbols)
                line_phones.append(word_phones)
                line_stresses.append(word_stresses)

        line_word_syms.append(self.special_word_symbols['<eos>'])
        line_char_syms.append(np.array([self.special_char_symbols['<eos>']]))
        line_phones.append(np.array([self.special_char_symbols['<eos>']]))
        line_stresses.append(np.array([self.special_char_symbols['<eos>']]))
        return line_word_syms, line_char_syms, line_phones, line_stresses, None, False

    def _get_dict_words(self, word):
        trans = [' ']*len(string.punctuation)
        trans[6] = "'"
        trans[11] = ","
        trans[13] = "."
        punc = string.punctuation
        # UTF8 nothing chars
        punc += ''.join([chr(i) for i in xrange(32)])
        punc += ''.join([chr(i) for i in xrange(127,160)])
        trans.extend([' ']*(len(punc)-len(trans)))
        trans = ''.join(trans)
        replace_punctuation = string.maketrans(punc, trans)

        translation = word.replace('\xe2\x80\x99',"'")\
                          .replace('\xd1\x81', 'c')\
                          .replace('\xe2\x80\xa6', '')\
                          .lower()\
                          .translate(replace_punctuation)\
                          .split()
        trans = {
            'bud1': 'bud'
        }
        dwords = []
        syms = []
        unk_word_int = [self.special_word_symbols['<unk>']]
        for w in translation:
            if w in trans:
                w = trans[w]
            # TODO: Might want to check on what words aren't in the word_to_intlist
            int_dwords = self.word_to_intlist.get(w, unk_word_int)
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
            lines = f.readlines()
            # add end of song
            lines.append('<eor>\n')
            for line in lines:
                # remove \n
                line = line[:-1]
                word_symbols, char_symbols, phones, stresses, rap_vecs, eov = self._process_line(line)

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
                    seq, labels, context = self.make_feature_dict(verse_word_symbols,
                                                                  verse_char_symbols,
                                                                  verse_phones,
                                                                  verse_stresses,
                                                                  verse_rap_vecs,
                                                                  as_lm=True)
                    yield seq, labels, context
                    verse_word_symbols = []
                    verse_char_symbols = []
                    verse_phones = []
                    verse_stresses = []
                    verse_rap_vecs = None

    def make_feature_dict(self, words, chars, phones, stresses, rap_vecs, as_lm=True):
        if as_lm:
            chars = chars[:-1]
            phones = phones[:-1]
            stresses = stresses[:-1]
            words = words[1:]
        features = {
                "chars": chars,
                "phones": phones,
                "stresses": stresses,
        }
        labels = words
        assert len(rap_vecs), "Attempt to make a feature without any rappers"
        context = {"rapper0": rap_vecs[0]}
        for r in xrange(1, self.max_nrps):
            if len(rap_vecs) > r:
                context["rapper{}".format(r)] = rap_vecs[r]
            else:
                context["rapper{}".format(r)] = np.zeros(self.len_rapper_vector)
        return features, labels, context

    def gen_features_from_starter(self, rappers, starter):
        words = []
        chars = []
        phones = []
        stresses = []
        rap_vecs = [self.rapper_vectors[r] for r in rappers]
        for line in starter:
            lwords, lchars, lphones, lstresses, lrap_vecs, leov =\
                    self._process_line(line)
            words.extend(lwords)
            chars.extend(lchars)
            phones.extend(lphones)
            stresses.extend(lstresses)
        features, labels, context = self.make_feature_dict(words,
                                                           chars,
                                                           phones,
                                                           stresses,
                                                           rap_vecs,
                                                           as_lm=False)
        actual_input_data = {"labels": np.array(labels)[np.newaxis, :]}
        verse_length = np.array([[len(features.values()[0])]])
        actual_input_data["verse_length"] = verse_length
        for key, array in features.iteritems():
            actual_input_data[key+".shape"] = np.array([[1, len(array)]])
            lengths = np.array([len(a) for a in array])[np.newaxis, :]
            actual_input_data[key+".lengths"] = lengths
            longest = max(lengths[0])
            padded = np.zeros((1, len(array), longest))
            for i, v in enumerate(array):
                padded[0, i, :len(v)] = v
            actual_input_data[key] = padded

        for key, array in context.iteritems():
            expanded = np.tile(array[np.newaxis, :], [verse_length[0,0], 1])
            actual_input_data[key] = expanded[np.newaxis, :]

        verse_ex = self.make_verse_instance(features, labels, context)
        tensor_dict, init_op_local = self.read_and_decode_single_example(verse_length, from_example=verse_ex)
        # Add a batch of 1
        for key, value in tensor_dict.iteritems():
            tensor_dict[key] = tf.expand_dims(value, 0)
        return tensor_dict, actual_input_data, init_op_local

    def update_rap_vectors(self, rap_sym, position):
        # add batch of 1
        input_data = {}
        zero_vec = np.zeros((1, 1, self.len_rapper_vector))
        if rap_sym in self.sym2char:
            rapper = self.sym2char[rap_sym]
            if rapper in self.rapper_vectors:
                # add batch of 1
                rap_vec = self.rapper_vectors[rapper][np.newaxis, np.newaxis, :]
                input_data["rapper"+str(position)] = rap_vec
                # reset rest of rappers
                for pos in xrange(position + 1, self.max_nrps):
                    input_data["rapper"+str(pos)] = zero_vec
        return input_data

    def tensorize(self, data, add_batch=True, name=None):
        if add_batch:
            data = data[np.newaxis, :]
        return tf.convert_to_tensor(data, name=name, dtype=tf.int32)

    def update_features(self, sym):
        input_data = {}
        word = self.int_to_word[sym]
        if word in self.special_word_symbols:
            char_feat = np.array([[[self.special_char_symbols[word]]]])
            input_data["chars"] = char_feat
            input_data["chars.lengths"] = np.array([[1]])
            #input_data["chars.shape"] = np.array([[1, 1]])
            input_data["phones"] = char_feat
            input_data["phones.lengths"] = np.array([[1]])
            #input_data["phones.shape"] = np.array([[1, 1]])
            input_data["stresses"] = char_feat
            input_data["stresses.lengths"] = np.array([[1]])
            #input_data["stresses.shape"] = np.array([[1, 1]])
        else:
            chars = [self.char2sym[c] for c in word]
            phones, stresses = self._extract_phones(sym)
            input_data["chars"] = np.array([[chars]])
            input_data["chars.lengths"] = np.array([[len(chars)]])
            #input_data["chars.shape"] = np.array([[1, len(chars)]])
            input_data["phones"] = np.array([[phones]])
            input_data["phones.lengths"] = np.array([[len(phones)]])
            #input_data["phones.shape"] = np.array([[1, len(phones)]])
            input_data["stresses"] = np.array([[stresses]])
            input_data["stresses.lengths"] = np.array([[len(stresses)]])
            #input_data["stresses.shape"] = np.array([[1, len(stresses)]])
        input_data["verse_length"] = np.array([[1]])
        input_data["labels"] = np.array([[sym]])
        return input_data

    def gen_context_from_nrp(self, prev_context, nrp_sym, position):
        if nrp_sym in self.sym2char:
            rapper = self.sym2char[nrp_sym]
            rap_vec = self.rapper_vectors[rapper]
            prev_context["rapper"+str(position)] = tf.cast(tf.tensor(rap_vec), tf.int32)
        return prev_context

    def load_verses(self, ftype):
        if ftype == 'train':
            filenames = self.train_filenames
        elif ftype == 'valid':
            filenames = self.valid_filenames
        else:
            filenames = self.test_filenames

        for f in filenames:
            for verse in self._load_file_verses(f):
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
            "phones.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "stresses.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "chars.shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "verse_length": tf.FixedLenFeature([1], dtype=tf.int64)
        }
        for r in xrange(self.max_nrps):
            context_features["rapper" + str(r)] = tf.FixedLenFeature([self.len_rapper_vector], dtype=tf.int64)

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

        # num_split = tf.ceil(length / max_num_steps)

        # context_keys = context_features.keys()
        # seq_keys = ["phones", "stresses", "chars", "labels"]
        # other_keys = [s+".lengths" for s in seq_keys[:-1]]

        # new_examples = [[casted_tensors[k] for k in context_features]
                        # for i in xrange(num_split)]
        # for seq in seq_keys + other_keys:
            # original = casted_tensors[seq]
            # split = tf.split(0, num_split, original)
            # for i, s in enumerate(split):
                # new_examples[i].append(s)
        # step_queue = tf.FIFOQueue(max_num_steps, tf.int32)
        # step_queue.enqueue_many(new_examples)
        # single_example = step_queue.dequeue()

        # key_ordering = context_keys + seq_keys + other_keys
        # init_op_local = tf.initialize_local_variables()
        # return key_ordering, single_example, init_op_local

    def cast_tensors(self, context, sequence):
        casted = {}
        for key, tensor in context.iteritems():
            if not key.endswith('shape'):
                casted[key] = tf.cast(tensor, tf.int32)

        for key, tensor in sequence.iteritems():
            ct = tf.cast(tensor, tf.int32)
            if key.endswith('lengths'):
                casted[key] = ct
            elif key + ".shape" in context:
                shape = tf.cast(context[key + ".shape"], tf.int32)
                casted[key] = tf.reshape(ct, shape)
            else:
                casted[key] = ct
        return casted

    def _split_phone_stress(self, phone_stress):
        try:
            stress = int(phone_stress[-1])
        except:
            return phone_stress, 1
        else:
            if stress > 2:
                pdb.set_trace()
            return phone_stress[:-1], stress

    def _extract_phones(self, wordint):
        phone_stresses = self._find_prons(wordint)

        word_phones = np.zeros(len(phone_stresses))
        word_stresses = np.zeros(len(phone_stresses))
        for i, phone_stress in enumerate(phone_stresses):
            phone, stress = self._split_phone_stress(phone_stress)
            try:
                word_phones[i] = self.phone2sym[phone]
            except:
                pdb.set_trace()
            try:
                word_stresses[i] = self.stress2sym[stress]
            except:
                pdb.set_trace()
        return word_phones, word_stresses

    def _find_prons(self, wordint):
        # word_int_to_prons should contain both slang and non-slang
        return self.word_int_to_pron[wordint]


if __name__ == '__main__':
    filenames = all_filenames("data/lyric_files/Tyler, The Creator")
    #filenames = all_filenames("data/lyric_files")
    train_ratio = .7
    valid_ratio = .2
    train_indices = np.random.choice(np.arange(len(filenames)),
                                     replace=False,
                                     size=int(len(filenames)*train_ratio))

    train_filenames = [f for i, f in enumerate(filenames) if i in train_indices]
    valid_test_filenames = [f for i, f in enumerate(filenames) if i not in train_indices]

    new_valid_ratio = valid_ratio / (1 - train_ratio)
    valid_indices = np.random.choice(np.arange(len(valid_test_filenames)),
                                     replace=False,
                                     size=int(len(valid_test_filenames)*new_valid_ratio))
    valid_filenames = [f for i, f in enumerate(valid_test_filenames) if i in valid_indices]
    test_filenames = [f for i, f in enumerate(valid_test_filenames) if i not in valid_indices]

    print len(train_filenames)
    print len(valid_filenames)
    print len(test_filenames)
    extractor = RapFeatureExtractor(train_filenames=train_filenames,
                                    valid_filenames=valid_filenames,
                                    test_filenames=test_filenames,
                                    from_config=False,
                                    config_file='data/config_new.p')
    extractor.save_tf_record_files(train_file="data/tf_train_data_new.txt",
                                   valid_file="data/tf_valid_data_new.txt",
                                   test_file="data/tf_test_data_new.txt")
