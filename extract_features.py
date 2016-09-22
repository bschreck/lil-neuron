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
from itertools import izip_longest
import pdb
EOV = -2
EOP = -3
EOS = -4
# NRP = -5
#TODO: Also include EOV (end of verse), NRP (new rapper),
# NRP is followed by an integer (which becomes an embedding)
# representing the rapper who is rapping the current verse

#TODO: all these are simple loops and should be in Cython

class Feature(object):
    def get_feat_name(self):
        return self.__name__

class PhonemeFeature(Feature):
    def extract_basic(phonemes, len_song_words,
                      max_phones_per_word, max_prons_per_word,
                      phones=True):
        last_index = 1
        if phones:
            last_index = 0

        features = -1*np.ones((len_song_words, max_prons_per_word))
        cur_word_i = 0
        for i, phrase in enumerate(phonemes):
            for j, word in enumerate(phrase):
                word_vector = -1*np.ones(max_phones_per_word*max_prons_per_word)
                for k, pron in enumerate(word):
                    start = k * max_phones_per_word
                    for i, phone_stress in enumerate(pron):
                        if phone_stress == -1:
                            break
                        stress = phone_stress[last_index]
                        word_vector[start + i] = stress
                features[cur_word_i] = word_vector
                cur_word_i += 1
            features[cur_word_i, 0] = EOP
            cur_word_i += 1

        assert cur_word_i == len_song_words - 1
        feature_vector[-1, 0] = EOS
        return features

class WordFeature(Feature):
    pass

class Stresses(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        return super(Stresses, self).extract_basic(phonemes, len_song_words,
                                                   max_phones_per_word, max_prons_per_word,
                                                   phones=False)

class RawPhonemes(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        return super(Stresses, self).extract_basic(phonemes, len_song_words,
                                                   max_phones_per_word, max_prons_per_word,
                                                   phones=True)

class WordGroupRhymeScheme(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class AssonanceRhyme(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass
class ConsonanceRhyme(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass
class MultiSyllabicRhyme(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class Alliteration(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class RawWords(WordFeature):
    def extract(self, song, song_dim):
        feature_vector = -1*np.ones((song_dim, 1))
        song_index = 0
        for v, verse in enumerate(song):
            for p, phrase in enumerate(verse):
                for w, word in enumerate(phrase):
                    if word == -1:
                        continue
                    feature_vector[song_index, 0] = word
                    song_index += 1
                feature_vector[song_index, 0] = EOP
                song_index += 1
            feature_vector[song_index, 0] = EOV
            song_index += 1
        assert song_index == song_dim - 1
        feature_vector[-1, 0] = EOS
        return feature_vector

class WordRhymeScheme(WordFeature):
    """
    Find rhymes by greedily searching ahead for words up to
    num_phrases forward for a rhyme, and
    and labeling words found with either a rhyme, or -1 if no
    rhyme is found.
    """
    def __init__(self, num_phrases=4):
        self.num_phrases = num_phrases
    def extract(self, song, song_dim):
        feature_vector = -1*np.ones((song_dim, 1))
        rhymes = song[...]
        max_current_rhyme = [-1]

        song_index = 0
        for p, phrase in enumerate(song):
            for w, word in enumerate(phrase):
                if word == -1:
                    continue
                self.find_and_label_rhymes(word, p, w,
                                           song[p:p + self.num_phrases, :],
                                           rhymes,
                                           max_current_rhyme)
                feature_vector[song_index, 0] = rhymes[p, w]
                song_index += 1
            feature_vector[song_index, 0] = EOP
            song_index += 1
        assert song_index == song_dim - 1
        feature_vector[-1, 0] = EOS
        return feature_vector

    def find_and_label_rhymes(self, word_int, p, w, to_search, rhymes,
                              max_current_rhyme):
        word = self.int_to_word[word_int]
        prev_max_rhyme_num = max_current_rhyme[0]
        current_rhyme_num = -1
        if rhymes[p, w] > -1:
            current_rhyme_num = rhymes[p, w]
        for new_p, phrase in enumerate(to_search):
            if new_p == 0:
                start_from = w + 1
            else:
                start_from = 0
            for pw, potential_word_int in enumerate(phrase[start_from:]):
                if potential_word_int == -1:
                    continue
                potential_word = self.int_to_word[potential_word_int]
                if potential_word in pronouncing.rhymes(word):
                    if current_rhyme_num > -1:
                        rhymes[p + new_p, start_from + pw] = current_rhyme_num
                    else:
                        max_current_rhyme[0] += 1
                        rhymes[p + new_p, start_from + pw] = max_current_rhyme[0]
                        rhymes[p, w] = max_current_rhyme[0]
                        current_rhyme_num = max_current_rhyme[0]
        assert max_current_rhyme[0] <= prev_max_rhyme_num + 1


class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
    phone_to_int = {p: i for i, p in enumerate(all_phones)}
    word_to_int = {}
    int_to_word = {}

    def __init__(self, train_filenames=None,
                       valid_filenames=None,
                       batch_size=3,
                       model_word_len=3,
                       chard2sym = {},
                       sym2char = {},
                       vocab_length =[0],
                       feature_set = None,
                       gzipped=True
                    ):
        self.train_filenames = train_filenames
        self.valid_filenames = valid_filenames
        self.lyric_iterator = lyric_iterator
        self.batch_size = batch_size
        self.model_word_len = model_word_len
        self.chard2sym = chard2sym
        self.sym2char = sym2char
        self.vocab_length = vocab_length

        self.train_char_syms = []
        self.train_phone_syms = []
        self.train_stress_syms = []
        self.valid_char_syms = []
        self.valid_phone_syms = []
        self.valid_stress_syms = []

        if feature_set:
            self.feature_set = feature_set
        else:
            self.feature_set = [Stresses(),
                                RawPhonemes(),
                                WordsAsChars()]
    def _process_line(line):
        line = line.lstrip().rstrip()
        words = line.split(" ")
        # TODO: Parse NRP
        if len(words) == 1 and words[0] == '':
            return ([self.eov_vector()],
                    [self.eov_vector()],
                    [self.eov_vector()])
        if len(words) == 1 and words[0] in self.special_symbols:
            return ([self.special_symbols(words[0])],
                    [self.special_symbols(words[0])],
                    [self.special_symbols(words[0])])
        line_syms = []
        line_phones = []
        line_stresses = []
        for i, word in enumerate(words):
            word_symbols = np.zeros(len(word),dtype=np.int32)
            word_phones, word_stresses = self._extract_phones(word)
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
            line_syms.extend(word_symbols)
            line_phones.extend(word_phones)
            line_stresses.extend(word_stresses)
        line_syms.append(self.eos_vector())
        line_phones.append(self.eos_vector())
        line_stresses.append(self.eos_vector())
        return line_syms, line_phones, line_stresses
    def _load_file(self, fname, valid=False):
        if self.gzipped:
            open_func = gzip.open
        else:
            open_func = open
        all_symbols = []
        all_phones = []
        all_stresses = []
        with open_func(filename, 'rb') as f:
            for line in f:
                symbols, phones, stresses = self._process_line(line))
                all_symbols.extend(symbols)
                all_phones.extend(phones)
                all_stresses.extend(stresses)
        if valid:
            self.valid_char_syms.extend(all_symbols)
            self.valid_phone_syms.extend(all_phones)
            self.valid_stress_syms.extend(all_stresses)
        else:
            self.train_char_syms.extend(all_symbols)
            self.train_phone_syms.extend(all_phones)
            self.train_stress_syms.extend(all_stresses)

    def _load_data(self):
        for f in self.train_filenames:
            self._load_file(f)
        for f in self.train_filenames:
            self._load_file(f, valid=True)
        self._standardize_word_lengths()
        self._calc_num_batches()

    def _reorder_data(x_in):
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

    def extract(self):
        self._load_data()
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        for f in self.feature_set:
            if f.ftype = 'phone':
                data = [self.train_phone_syms, self.valid_phone_syms]
            elif f.ftype = 'stress':
                data = [self.train_stress_sym, self.valid_stress_syms]
            else:
                data = [self.train_char_syms, self.valid_char_syms]
            unordered_t, unordered_v = f.extract(*data)
            x_t, y_t = self._reorder_data(unordered_t)
            x_v, y_v = self._reorder_data(unordered_v)
            x_train.append(x_t)
            y_train.append(y_t)
            x_valid.append(x_v)
            y_valid.append(y_v)
        return x_train, y_train, x_valid, y_valid


    @classmethod
    def find_shape(cls, seq):
        try:
            len_ = len(seq)
        except TypeError:
            return ()
        shapes = [cls.find_shape(subseq) for subseq in seq]
        return (len_,) + tuple(max(sizes) for sizes in izip_longest(*shapes,
                                                                    fillvalue=1))

    @classmethod
    def fill_array(cls, arr, seq):
        if arr.ndim == 1:
            try:
                len_ = len(seq)
            except TypeError:
                len_ = 0
            arr[:len_] = seq
        else:
            for subarr, subseq in izip_longest(arr, seq, fillvalue=()):
                cls.fill_array(subarr, subseq)

    @classmethod
    def phone_int_encode(cls, phone):
        return cls.phone_to_int[phone]

    @classmethod
    def word_int_encode(cls, word):
        if word in cls.word_to_int:
            return cls.word_to_int[word]
        else:
            n = len(cls.word_to_int)
            cls.word_to_int[word] = n
            cls.int_to_word[n] = word
            return n

    @classmethod
    def split_phone_stress(cls, phone_stress):
        try:
            stress = int(phone_stress[-1])
        except:
            return phone_stress, 0
        else:
            return phone_stress[:-1], stress

    def feature_set(self):
        return self.phone_feature_set + self.word_feature_set

    def extract_features_to_file(self):
        for song in self.lyric_iterator:
            output = self.extract_phonemes(song)
            song_as_phonemes = output[0]
            song_as_ints = output[1]
            len_song_words = output[2]
            max_phones_per_word = output[3]
            max_prons_per_word = output[4]
            # encode phones as vector of size max_phones_per_word*max_pronunciations_per_word
            phone_feature_vector_shape = [len_song_words, max_phones_per_word*max_prons_per_word]
            # will be one-hot-encoded but for now just use a single number
            word_feature_vector_shape = [len_song_words, 1]


            shape_phone_features = tuple([len(self.phone_feature_set)] + phone_feature_vector_shape)
            song_phone_features = -1 * np.ones(shape_phone_features, dtype=np.int32)
            for i, feature_extractor in enumerate(self.phone_feature_set):
                feature = feature_extractor.extract(song_as_phonemes,
                                                    len_song_words,
                                                    max_phones_per_word,
                                                    max_prons_per_word)
                if feature is not None:
                    song_phone_features[i, :] = feature

            shape_word_features = tuple([len(self.word_feature_set)] + word_feature_vector_shape)
            song_word_features = -1 * np.ones(shape_word_features, dtype=np.int32)
            for i, feature_extractor in enumerate(self.word_feature_set):
                feature = feature_extractor.extract(song_as_ints, len_song_words)
                if feature is not None:
                    song_word_features[i, :] = feature
            self.serialize_and_write_to_disk(song_word_features, song_phone_features)

    def extract_phonemes(self, song):
        # TODO: change to extract_phonemes for a word
        phones = []
        words = []
        len_max_verse = 0
        len_max_phrase = 0
        len_song_words = 0
        max_phones_per_word = 0
        max_prons_per_word = 0
        for num_verse, verse in enumerate(song):
            phones.append([])
            verse_words = []
            for num_phrase, phrase in enumerate(verse):
                phones[-1].append([])
                phrase_words = []
                for word in phrase.split(' '):
                    if word == '':
                        continue
                    phrase_words.append(self.word_int_encode(word))
                    pronunciations = []
                    for pr in pronouncing.phones_for_word(word):
                        word_phone_stresses = []
                        for phone_stress in pr.split(' '):
                            phone, stress = self.split_phone_stress(phone_stress)
                            word_phone_stresses.append([self.phone_int_encode(phone), stress])
                        len_phones = len(word_phone_stresses)
                        if len_phones > max_phones_per_word:
                            max_phones_per_word = len_phones
                        pronunciations.append(word_phone_stresses)
                    phones[-1][-1].append(pronunciations)
                    len_pron = len(pronunciations)
                    if len_pron > max_prons_per_word:
                        max_prons_per_word = len_pron
                len_phrase = len(phrase_words)
                if len_phrase > len_max_phrase:
                    len_max_phrase = len_phrase

                verse_words.append(phrase_words)
                # add number of words plus <end-of-phrase>
                len_song_words += (len_phrase + 1)
            len_verse = num_phrase + 1
            if len_verse > len_max_verse:
                len_max_verse = len_verse
            words.append(verse_words)
            # add <end-of-verse>
            len_song_words += 1
        # add <end-of-song>
        len_song_words +=1
        phones_arr = -1 * np.ones(self.find_shape(phones), dtype=np.int64)
        self.fill_array(phones_arr, phones)

        num_verses = num_verse + 1
        words_arr = -1 * np.ones((num_verses, len_max_verse, len_max_phrase), dtype=np.int64)
        self.fill_array(words_arr, words)
        return phones_arr, words_arr, len_song_words, max_phones_per_word, max_prons_per_word


    def serialize_and_write_to_disk(self, song_word_features, song_phone_features):
        print "="*50, "SONG", "="*50
        for i, feat in enumerate(song_word_features):
            feat_name = self.feature_set[i].get_feat_name()
            print "------FEATURE {}------",format(feat_name)
            print feat

