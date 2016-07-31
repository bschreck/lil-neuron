"""
Pronunciation features:
    - phonemes as stresses
    - raw phonemes, no stresses
    - raw phonemes, broken up into groups separated by stresses
    - raw phonemes, broken up into groups separated by words
    - raw phonemes, broken up into groups separated by phrases or lines
    - rhyme scheme as categorical variables (mark each rhyme with same category)
"""
import pronouncing as pron
import numpy as np
from itertools import izip_longest


class RapFeatureExtractor(object):
    all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                      "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                      "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                      "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
    phone_to_int = {p: i for i, p in enumerate(all_phones)}
    word_to_int = {}
    int_to_word = {}

    def __init__(self, lyric_iterator):
        self.lyric_iterator = lyric_iterator
        self.phone_feature_set = [self.stresses,
                                  self.raw_phonemes,
                                  self.stressed_phonemes,
                                  self.word_phonemes,
                                  self.phrase_phonemes,
                                  self.rhyme_scheme]
        self.word_feature_set = [self.raw_words]

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

    def extract_features_to_file(self):
        for song in self.lyric_iterator:
            song_as_phonemes, song_as_ints = self.extract_phonemes(song)

            shape_song_phones = song_as_phonemes.shape
            shape_phone_features = tuple([len(self.phone_feature_set)] + list(shape_song_phones[:-1]))
            song_phone_features = -1 * np.ones(shape_phone_features, dtype=np.int64)
            # for i, feature_extractor in enumerate(self.phone_feature_set):
                # feature = feature_extractor(song_as_phonemes)
                # song_phone_features[i, :] = feature

            shape_song_words = song_as_ints.shape
            shape_word_features = tuple([len(self.word_feature_set)] + list(shape_song_words))
            song_word_features = -1 * np.ones(shape_word_features, dtype=np.int64)
            for i, feature_extractor in enumerate(self.word_feature_set):
                feature = feature_extractor(song_as_ints)
                song_word_features[i, :] = feature
            self.serialize_and_write_to_disk(song_word_features, song_phone_features)

    def extract_phonemes(self, song):
        phones = []
        words = []
        len_max_phrase = 0
        len_song = 0
        for phrase in song:
            len_song += 1
            phones.append([])
            phrase_words = []
            for word in phrase.split(' '):
                phrase_words.append(self.word_int_encode(word))
                pronunciations = []
                for pr in pron.phones_for_word(word):
                    word_phone_stresses = []
                    for phone_stress in pr.split(' '):
                        phone, stress = self.split_phone_stress(phone_stress)
                        word_phone_stresses.append([self.phone_int_encode(phone), stress])
                    pronunciations.append(word_phone_stresses)
                phones[-1].append(pronunciations)
            if len(phrase_words) > len_max_phrase:
                len_max_phrase = len(phrase_words)
            words.append(phrase_words)
        phones_arr = -1 * np.ones(self.find_shape(phones), dtype=np.int64)
        self.fill_array(phones_arr, phones)

        words_arr = -1 * np.ones((len_song, len_max_phrase), dtype=np.int64)
        self.fill_array(words_arr, words)
        print words_arr
        return phones_arr, words_arr


    def serialize_and_write_to_disk(self, song_word_features, song_phone_features):
        print "="*50, "SONG", "="*50
        for feat in song_word_features:
            print "------FEATURE------"
            print feat

    def raw_words(self, song):
        return song

    def stresses(self, phonemes):
        phone_shape = list(phonemes.shape)
        just_stresses_shape = phone_shape[:-1] + [1]
        return phonemes[just_stresses_shape]

    def raw_phonemes(self, phonemes):
        phone_shape = list(phonemes.shape)
        just_phones_shape = phone_shape[:-1] + [0]
        return phonemes[just_phones_shape]

    def stressed_phonemes(self, phonemes):
        pass
    def word_phonemes(self, phonemes):
        pass
    def phrase_phonemes(self, phonemes):
        pass
    def rhyme_scheme(self, phonemes):
        pass
