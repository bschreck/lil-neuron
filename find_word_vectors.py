from generate_lyric_files import all_filenames
from nltk.tokenize import StanfordTokenizer
import string
from collections import Counter
import re
import numpy as np


def load_glove_vectors(filename, word2int):
    array = np.zeros((max(word2int.values())+1, 300))
    with open(filename, 'r') as f:
        for line in f:
            vals = line.rstrip().split(' ')
            vector = [float(x) for x in vals[1:]]
            if array is None:
                array = np.zeros((max(word2int.values())+1, len(vector)))
            wordint = word2int.get(vals[0], None)
            if wordint:
                array[wordint] = vector
    return array

# def find_words():
    # filenames = all_filenames("data/lyric_files")
    # trans = [' ']*len(string.punctuation)
    # trans[6] = "'"
    # # trans[11] = ","
    # # trans[13] = "."
    # trans = ''.join(trans)
    # replace_punctuation = string.maketrans(string.punctuation, trans)
    # all_words = set()
    # for f in filenames:
        # with open(f, 'r') as fo:
            # words = fo.read()\
                    # .replace("<eov>", " ")\
                    # .replace("\n", " ")\
                    # .replace('\xe2\x80\x99',"'")\
                    # .lower()\
                    # .translate(replace_punctuation)\
                    # .split()
            # all_words |= set(words)
    # return all_words






def find_unknown(vectors, word_counts):
    unknown = [w for w in word_counts if w not in vectors]
    ing_words = set([w[0] for w in unknown if  w[0].endswith('in') and w[0]+'g' in vectors])
    unknown = [w for w in unknown if w not in ing_words]
    return list(unknown)

                # numbers = re.findall(num, word)
                # if numbers:
                    # number = numbers[0][0]
                    # nwords = _word_numbers(number)
                    # add_word(word, nwords)
                # else:
                    # cd_split = word.translate(commas_decimals).split()
                    # new_words = []
                    # nondict = []
                    # for i, w in enumerate(cd_split):
                        # if english_d.check(w):
                            # new_words.append(w)
                        # else:
                            # suggested = english_d.suggest(w)
                            # if len(suggested) and lev.distance(w, suggested[0]) == 1:
                                # new_words.append(suggested[0])
                            # else:
                                # new_words.append(w)
                                # nondict.append(i)

# if __name__ == '__main__':
    # vectors =
    # unknown = find_word_vectors(vectors)

