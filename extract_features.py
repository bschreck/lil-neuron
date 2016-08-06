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
EOP = -2
EOS = -3
#TODO: Also include EOV (end of verse), NRP (new rapper),
# NRP is followed by an integer (which becomes an embedding)
# representing the rapper who is rapping the current verse

class Feature(object):
    def get_feat_name(self):
        return self.__name__

class PhonemeFeature(Feature):
    pass

class WordFeature(Feature):
    pass

class Stresses(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        # TODO: serialize phones into feature vector
        return phonemes[...,1]

class RawPhonemes(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        # TODO: serialize phones into feature vector
        return phonemes[...,0]

class StressedPhonemes(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class WordPhonemes(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class PhrasePhonemes(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

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
class MultiSyllabicRhyme(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass
class Alliteration(PhonemeFeature):
    def extract(self, phonemes, len_song_words,
                max_phones_per_word, max_prons_per_word):
        pass

class RawWords(WordFeature):
    def extract(self, words, song_dim):
        # TODO: serialize words into feature vector
        return words

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
        assert song_index == song_dim
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
                if potential_word in pron.rhymes(word):
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

    def __init__(self, lyric_iterator):
        self.lyric_iterator = lyric_iterator
        self.phone_feature_set = [Stresses(),
                                  RawPhonemes(),
                                  StressedPhonemes(),
                                  WordPhonemes(),
                                  PhrasePhonemes(),
                                  WordGroupRhymeScheme()]
        self.word_feature_set = [RawWords(), WordRhymeScheme(num_phrases=4)]

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
        phones = []
        words = []
        len_max_phrase = 0
        len_song_words = 0
        max_phones_per_word = 0
        max_prons_per_word = 0
        for num_phrase, phrase in enumerate(song):
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
                    len_phones = len(word_phone_stresses)
                    if len_phones > max_phones_per_word:
                        max_phones_per_word = len_phones
                    pronunciations.append(word_phone_stresses)
                phones[-1].append(pronunciations)
                len_pron = len(pronunciations)
                if len_pron > max_prons_per_word:
                    max_prons_per_word = len_pron
            len_phrase = len(phrase_words)
            if len_phrase > len_max_phrase:
                len_max_phrase = len_phrase

            words.append(phrase_words)
            # add number of words plus <end-of-phrase>
            len_song_words += (len_phrase + 1)
        phones_arr = -1 * np.ones(self.find_shape(phones), dtype=np.int64)
        self.fill_array(phones_arr, phones)

        # num phrases plus <end-of-song>
        num_phrases = num_phrase + 2
        words_arr = -1 * np.ones((num_phrases, len_max_phrase), dtype=np.int64)
        self.fill_array(words_arr, words)
        return phones_arr, words_arr, len_song_words, max_phones_per_word, max_prons_per_word


    def serialize_and_write_to_disk(self, song_word_features, song_phone_features):
        print "="*50, "SONG", "="*50
        for i, feat in enumerate(song_word_features):
            feat_name = self.feature_set[i].get_feat_name()
            print "------FEATURE {}------",format(feat_name)
            print feat

