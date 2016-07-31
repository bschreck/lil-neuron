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
import pdb


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
                                  self.word_group_rhyme_scheme]
        self.word_feature_set = [self.raw_words, self.word_rhyme_scheme]

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
            for i, feature_extractor in enumerate(self.phone_feature_set):
                feature = feature_extractor(song_as_phonemes)
                if feature is not None:
                    song_phone_features[i, :] = feature

            shape_song_words = song_as_ints.shape
            shape_word_features = tuple([len(self.word_feature_set)] + list(shape_song_words))
            song_word_features = -1 * np.ones(shape_word_features, dtype=np.int64)
            for i, feature_extractor in enumerate(self.word_feature_set):
                feature = feature_extractor(song_as_ints)
                if feature is not None:
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

    def get_phone_feat_name(self, feat_index):
        return self.get_feat_name(self.phone_feature_set, feat_index)

    def get_word_feat_name(self, feat_index):
        return self.get_feat_name(self.word_feature_set, feat_index)

    def get_feat_name(self, feature_set, feat_index):
        return feature_set[feat_index].__name__.upper().replace('_', ' ')

    def serialize_and_write_to_disk(self, song_word_features, song_phone_features):
        print "="*50, "SONG", "="*50
        for i, feat in enumerate(song_word_features):
            feat_name = self.get_word_feat_name(i)
            print "------FEATURE {}------",format(feat_name)
            print feat

    def raw_words(self, song):
        return song

    def stresses(self, phonemes):
        return phonemes[...,1]

    def raw_phonemes(self, phonemes):
        return phonemes[...,0]

    def stressed_phonemes(self, phonemes):
        # ?
        pass
    def word_phonemes(self, phonemes):
        pass
    def phrase_phonemes(self, phonemes):
        pass
    def word_group_rhyme_scheme(self, phonemes):
        pass

    def word_rhyme_scheme(self, song, num_phrases=4):
        """
        Find rhymes by greedily searching ahead for words up to
        num_phrases forward for a rhyme, and
        and labeling words found with either a rhyme, or -1 if no
        rhyme is found.
        """
        rhymes = song[...]
        max_current_rhyme = -1
        for p, phrase in enumerate(song):
            for w, word in enumerate(phrase):
                if word == -1:
                    continue
                max_current_rhyme = self.find_and_label_rhymes(word, p, w,
                                                               song[p:p + num_phrases, :],
                                                               rhymes,
                                                               max_current_rhyme)
        return rhymes

    def find_and_label_rhymes(self, word_int, p, w, to_search, rhymes,
                              max_current_rhyme):
        word = self.int_to_word[word_int]
        prev_max_rhyme_num = max_current_rhyme
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
                if potential_word in pron.rhymes(word):
                    if current_rhyme_num > -1:
                        rhymes[p + new_p, start_from + pw] = current_rhyme_num
                    else:
                        max_current_rhyme += 1
                        rhymes[p + new_p, start_from + pw] = max_current_rhyme
                        rhymes[p, w] = max_current_rhyme
                        current_rhyme_num = max_current_rhyme
        assert max_current_rhyme <= prev_max_rhyme_num + 1
        return max_current_rhyme
