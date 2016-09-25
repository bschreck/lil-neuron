class Feature(object):
    def get_feat_name(self):
        return self.__name__

    def extract(self, *data):
        pass


class PhonemeFeature(Feature):
    pass


class StressFeature(Feature):
    pass


class WordFeature(Feature):
    pass


class RawStresses(StressFeature):
    def extract(self, *data):
        self.vector_dim = data[0].shape[-1]
        return data


class RawPhonemes(PhonemeFeature):
    def extract(self, *data):
        self.vector_dim = data[0].shape[-1]
        return data


class RawWordsAsChars(WordFeature):
    def extract(self, *data):
        self.vector_dim = data[0].shape[-1]
        return data


class WordGroupRhymeScheme(PhonemeFeature):
    pass


class AssonanceRhyme(PhonemeFeature):
    pass


class ConsonanceRhyme(PhonemeFeature):
    pass


class MultiSyllabicRhyme(PhonemeFeature):
    pass


class Alliteration(PhonemeFeature):
    pass


class WordRhymeScheme(WordFeature):
    """
    Find rhymes by greedily searching ahead for words up to
    num_phrases forward for a rhyme, and
    and labeling words found with either a rhyme, or -1 if no
    rhyme is found.
    """
    pass
