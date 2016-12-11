import cPickle as pickle


class SpellChecker(object):
    def __init__(self, counter_corpus_filename):
        with open(counter_corpus_filename, 'r') as f:
            self.word_counter = pickle.load(f)

    def P(self, word):
        "Probability of `word`."
        N = float(sum(self.word_counter.values()))
        return self.word_counter[word] / N

    def correction(self, word):
        "Most probable spelling correction for word."
        if word.endswith('s') and word[:-1] in self.word_counter:
            return None
        candidates1 = self.candidates1(word, self.word_counter)
        if len(candidates1):
            return max(candidates1, key=self.P)
        else:
            return None
            # candidates2 = self.candidates2(word, self.word_counter)
            # if len(candidates2):
                # return max(candidates2, key=self.P)
            # else:
                # return None

    def candidates1(self, word, word_counter):
        "Generate possible spelling corrections for word."
        return (self.known(self.vowels_and_ing(word)) or self.known(self.vowels_and_ing2(word)))

    def candidates2(self, word, word_counter):
        "Generate possible spelling corrections for word."
        return (self.known(self.edits1(word)) or self.known(self.edits2(word)))

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.word_counter)

    def vowels_and_ing(self, word, same_length=False):
        letters = 'aeiou'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        if same_length:
            deletes = []
        else:
            # only delete vowels, and don't delete end letters
            deletes = [L + R[1:] for L, R in splits[1:-2] if R and R[0] in letters]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        if same_length:
            inserts = []
        else:
            inserts = [L + c + R for L, R in splits for c in letters]

        if word.endswith('in'):
            inserts.append(word + 'g')
        return set(deletes + transposes + replaces + inserts)

    def vowels_and_ing2(self, word):
        return (e2 for e1 in self.vowels_and_ing(word) for e2 in self.vowels_and_ing(e1, same_length=True))

    def edits1(self, word, same_length=False):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        if same_length:
            deletes = []
        else:
            deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        if same_length:
            inserts = []
        else:
            inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1, same_length=True))
