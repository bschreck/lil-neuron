import extract_features as exf
#import unittest
import pstats, cProfile
import sys
import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                             reload_support=True)
#import pyximport; pyximport.install()
import c_extract_features as copt
from IPython import get_ipython
ipython = get_ipython()


class TestFeatureExtractor(object):#unittest.TestCase):
    def setUp(self):
        self.rap = [
            ["cold blooded killer",
            "mold loving  dealer"],

            ["bold hooded  biller",
            "old  oven"]
        ]
        self.lyric_iterator = [self.rap]
        self.extractor = exf.RapFeatureExtractor(self.lyric_iterator)

    def extract_phonemes(self, rap):
        output = self.extractor.extract_phonemes(rap)
        self.song_as_phonemes = output[0]
        self.song_as_ints = output[1]
        self.len_song_words = output[2]
        self.max_phones_per_word = output[3]
        self.max_prons_per_word = output[4]

    def test_raw_words(self):
        self.extract_phonemes(self.rap)
        self.assertEqual(self.len_song_words, (((3 + 1) * 2 + 1) * 2) + 1 - 1)
        self.assertEqual(self.max_phones_per_word, 6)
        self.assertEqual(self.max_prons_per_word, 1)
        shape = self.song_as_ints.shape

        self.assertEqual(len(shape), 3)
        self.assertEqual(shape[0], 2)
        self.assertEqual(shape[1], 2)
        self.assertEqual(shape[2], 3)

        raw_words = exf.RawWords()
        feature_vector = raw_words.extract(self.song_as_ints, self.len_song_words)
        self.assertEqual(feature_vector.shape[0], self.len_song_words)
        self.assertEqual(feature_vector.shape[1], 1)
        true_vector = [0,1,2,-3,3,4,5,-3,-2, 6, 7, 8,-3, 9,10,-3,-2,-4]
        for i, w in enumerate(feature_vector[:,0]):
            self.assertEqual(w, true_vector[i])

    def test_copt_raw_words(self):
        self.extract_phonemes(self.rap)
        raw_words = exf.RawWords()
        ipython.magic("timeit raw_words.extract(self.song_as_ints, self.len_song_words)")
        ipython.magic("timeit copt.extract_raw_words(self.song_as_ints, self.len_song_words)")

    def assertEqual(self, first, second):
        assert first == second, "{} != {}".format(first, second)

if __name__ == "__main__":
    test = TestFeatureExtractor()
    test.setUp()
    test.test_copt_raw_words()
