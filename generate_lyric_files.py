from pymongo import MongoClient
import time
import string
import os
import numpy as np
import cPickle as pck
import pdb
from collections import Counter, defaultdict
import re
from nltk.tokenize import StanfordTokenizer
from nltk.corpus.reader.cmudict import CMUDictCorpusReader
from spell_checker import SpellChecker
import inflect
inflect_engine = inflect.engine()


def mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)


def generate_files(dirname):
    rappers = db.artists.find({'albums': {'$exists': True}})
    with open('missing_rappers.txt', 'r') as f:
        missing = set([r.replace('\n', '') for r in f.readlines()])
    rappers = [r for r in rappers
               if r['name'] in missing]
    print len(rappers)
    for rap_i, rapper in enumerate(rappers):
        rapper_name = rapper["name"]
        albums = rapper['albums']
        if type(albums) != list:
            iterator = albums.iteritems()
        else:
            iterator = enumerate(albums)

        for album_i, album in iterator:
            album_name = album["name"]
            album_year = album["year"]
            songs = album['songs']
            unknown_song_index = 0

            if type(songs) != list:
                song_iterator = songs.iteritems()
            else:
                song_iterator = enumerate(songs)
            for song_i, song in song_iterator:
                song_name = song["name"]
                if "lyrics" not in song or not song["lyrics"]:
                    continue
                lyrics = song["lyrics"]
                filename = formulate_filename(dirname,
                                              rapper_name,
                                              album_year,
                                              album_name,
                                              song_name)
                if filename.endswith("unknown_song"):
                    filename += str(unknown_song_index)
                    unknown_song_index += 1
                mkdir_recursive(os.path.dirname(filename))
                with open(filename, 'w') as f:
                    if not lyrics[0][0].startswith('(NRP:'):
                        f.write(u'<nrp:{}>\n'.format(rapper_name).encode('utf8'))
                    else:
                        first_line = lyrics[0][0]
                        words = first_line.split('(NRP:')
                        rappers = [w.replace(")", "").strip().replace(' ', '_').lower()
                                   for w in words if w]
                        first_line = '<nrp:{}>\n<eos>\n'.format(';'.join(rappers))
                        lyrics[0][0] = first_line
                    for verse in lyrics:
                        f.write(u'\n'.join(verse).encode('utf8'))
                        f.write(u'\n<eov>\n'.encode('utf8'))


def formulate_filename(dirname, rapper, year, album, song):
    if not year:
        year = "unknown_year"
    if not album:
        album = "unknown_album"
    if not song:
        song = "unknown_song"
    return "/".join([dirname, rapper, str(year), album, song]) + ".txt"


def all_filenames(dirname):
    filenames = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            if file.endswith(".txt"):
                filenames.append(os.path.join(root, file))
    return filenames


def make_single_corpus_file(filenames, output_filename):
    def format_rapper(match):
        rapper = match.group(2).lower().replace(' ', '_')
        return '<nrp:{}>'.format(rapper)
    with open(output_filename, 'wb') as out:
        for f in filenames:
            with open(f, 'r') as fo:
                lines = fo.readlines()
                for line in lines:
                    # TODO: remove if I redo lyric files
                    if line.lstrip().startswith('(NRP:'):
                        words = line.split('(NRP:')
                        rappers = [w.replace(")", "").strip().replace(' ', '_').lower()
                                   for w in words if w]
                        line = '<nrp:{}>\n<eos>\n'.format(';'.join(rappers))
                    else:
                        line += '\n<eos>\n'
                    out.write(line)
                out.write('\n<eor>\n')


def word_numbers(num):
    num = num.replace(',', '')
    return inflect_engine.number_to_words(num).split()



def tokenize_and_save_corpus(corpus_filename, new_filename):
    with open(corpus_filename, 'r') as f:
        corpus_str = f.read()
    tokenized = StanfordTokenizer().tokenize(corpus_str)
    lowered = [w.lower() for w in tokenized]

    num = r'(?<!\S)(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)(?!\S)'
    number_words = {}
    new_words = []
    for word in lowered:
        if word in number_words:
            new_words.extend(number_words[word])
        else:
            numbers = re.findall(num, word)
            if numbers:
                number = numbers[0][0]
                nwords = word_numbers(number)
                number_words[word] = nwords
                new_words.extend(nwords)
            else:
                new_words.append(word)
    with open(new_filename, 'w') as f:
        f.write(' '.join(new_words).encode('utf-8'))


def create_correct_pronouncing_corpus(tokenized_corpus_filenames, new_filename,
                                      existing_slang_words=None):
    corpus = []
    for fname in tokenized_corpus_filenames:
        with open(fname, 'r') as f:
            corpus.extend(f.read().split())
    possible_words = set(CMUDictCorpusReader('data/pronouncing', 'cmudict').words())
    possible_words |= existing_slang_words
    correct = Counter((w for w in corpus if w in possible_words))
    with open(new_filename, 'w') as f:
        pck.dump(correct, f)


def build_word_dict(tokenized_corpus_filenames, pronouncing_word_count_filename):
    word_to_dword = {}
    dword_to_int = {}
    word_to_int = {}
    word_counts = defaultdict(int)
    slang_word_counts = defaultdict(int)

    def add_word(orig, dword=None):
        if not dword:
            word_to_int[orig] = len(dword_to_int)
            slang_word_counts[orig] += 1
            return

        if dword not in dword_to_int:
            dword_to_int[dword] = len(dword_to_int)
        if orig not in word_to_dword:
            word_to_dword[orig] = dword
            word_to_int[orig] = dword_to_int[dword]
        word_counts[dword] += 1

    spell_checker = SpellChecker(pronouncing_word_count_filename)
    for fname in tokenized_corpus_filenames:
        with open(fname, 'r') as fo:
            words = fo.read().split()
            for word in words:
                if word in spell_checker.word_counter:
                    add_word(word, word)
                else:
                    if len(word) > 5:
                        correction = spell_checker.correction(word)
                        if correction:
                            add_word(word, correction)
                        else:
                            add_word(word)
                    else:
                        add_word(word)
    return dict(word_to_dword=word_to_dword,
                dword_to_int=dword_to_int,
                word_to_int=word_to_int,
                word_counts=word_counts,
                slang_word_counts=slang_word_counts)


def train_test_split(dirname, train_ratio=.9, valid_ratio=.05):
    filenames = all_filenames(dirname)
    train_indices = np.random.choice(np.arange(len(filenames)),
                                     replace=False,
                                     size=int(len(filenames) * train_ratio))

    train_filenames = [f for i, f in enumerate(filenames) if i in train_indices]
    valid_test_filenames = [f for i, f in enumerate(filenames) if i not in train_indices]

    new_valid_ratio = valid_ratio / (1 - train_ratio)
    valid_indices = np.random.choice(np.arange(len(valid_test_filenames)),
                                     replace=False,
                                     size=int(len(valid_test_filenames) * new_valid_ratio))
    valid_filenames = [f for i, f in enumerate(valid_test_filenames) if i in valid_indices]
    test_filenames = [f for i, f in enumerate(valid_test_filenames) if i not in valid_indices]
    return train_filenames, valid_filenames, test_filenames


def process_corpi(output_dict_filename='data/word_dicts.p'):
    train, valid, test = train_test_split("data/lyric_files")
    corpus_filenames = []
    tokenized_filenames = []
    for ctype, filenames in zip(['valid', 'test', 'train'], [valid, test, train]):
        corpus_file = "data/{}_corpus.txt".format(ctype)
        corpus_filenames.append(corpus_file)
        make_single_corpus_file(filenames, corpus_file)
        tokenized_file = "data/{}_corpus_tokenized.txt".format(ctype)
        tokenized_filenames.append(tokenized_file)
        tokenize_and_save_corpus(corpus_file, tokenized_file)

    pronouncing_word_count_filename = 'data/pronouncing/word_count.p'

    existing_slang_words = set(get_existing_slang_words().keys())
    create_correct_pronouncing_corpus(tokenized_filenames, pronouncing_word_count_filename,
                                      existing_slang_words=existing_slang_words)

    dicts = build_word_dict(tokenized_filenames, pronouncing_word_count_filename)
    with open(output_dict_filename, 'wb') as f:
        pck.dump(dicts, f)

    # run load_slang_words_into_mongo
    # Run find pronuncations on slang_word_counts in dicts
    # Should be able to look up GloVe vectors for most words in both slang and dictionary word_counts
    return dicts


def get_existing_slang_words():
    from pymongo import MongoClient
    client = MongoClient()
    db = client['lil-neuron-db']
    slang_words = {}
    records = db.slang_words.find()
    for r in records:
        if 'pronunciation' in r:
            slang_words[r['word']] = r['pronunciation']
    return slang_words

def load_slang_words_into_mongo(word_dict_file):
    from pymongo import MongoClient
    client = MongoClient()
    db = client['lil-neuron-db']
    with open(word_dict_file, 'r') as f:
        dicts = pck.load(f)
        slang_words = dicts['slang_word_counts']
    pronunciations = CMUDictCorpusReader('data/pronouncing', 'cmudict').dict()
    for w, c in slang_words.iteritems():
        if '--' in w or ',' in w:
            continue
        else:
            if not db.slang_words.find_one({'word': w}):
                word_dict = {'word': w, 'count': c}
                if w.endswith('in') and w + 'g' in pronunciations:
                    pron = pronunciations[w + 'g'][0]
                    word_dict['pronunciation'] = pron
                elif w.endswith('ins') and w[:-1] + 'g' in pronunciations:
                    pron = pronunciations[w[:-1] + 'g'][0] + ['Z']
                    word_dict['pronunciation'] = pron
                elif w.endswith('s') and w[:-1] in pronunciations:
                    pron = pronunciations[w[:-1]][0] + ['Z']
                    word_dict['pronunciation'] = pron
                db.slang_words.insert_one(word_dict)

def load_all_pronunciations():
    pronunciations = CMUDictCorpusReader('data/pronouncing', 'cmudict').dict()
    from pymongo import MongoClient
    client = MongoClient()
    db = client['lil-neuron-db']
    records = db.slang_words.find({'pronunciation': {'$exists': True}})
    for r in records:
        pronunciations[r['word']] = r['pronunciation']
    return pronunciations

def load_words_and_prons(tokenized_corpus_filename, glove_vector_fname):
    with open(tokenized_corpus_filename, 'r') as f:
        corpus = f.read().split()
    vectors = load_glove_vectors(glove_vector_fname)
    pronunciations = load_all_pronunciations()
    word_vectors = []
    word_prons = []
    for w in corpus:
        #check special symbols here
        number_convert = re.sub('\d', '#', w)
        #use number_convert go get vectors
        # In future can try a <unk> vector and pron
        unk_vector = False
        if number_convert in vectors and w in pronunciations:
            word_vectors.append(vectors[number_convert])
            word_prons.append(pronunciations[w])

# # TODO: acronyms from extract_features._find_prons
# # TODO: manually get pronunciations for most common slang words in corpus
# def build_word_dict(filenames):
    # begin = time.time()
    # trans = [' ']*len(string.punctuation)
    # trans[6] = "'"
    # trans[11] = ","
    # trans[13] = "."
    # trans = ''.join(trans)
    # replace_punctuation = string.maketrans(string.punctuation, trans)
    # commas_decimals = string.maketrans(',.', '  ')
    # num = r'(?<!\S)(\d*\.?\d+|\d{1,3}(,\d{3})*(\.\d+)?)(?!\S)'

    # dword_to_int = {}
    # int_to_dword = {}
    # word_to_dwordint = {}
    # word_counts = defaultdict(int)
    # slang_word_counts = defaultdict(int)
    # def add_word(orig, wlist, nondict=[]):
        # if orig not in word_to_dwordint:
            # wintlist = []
            # for w in wlist:
                # if w in dword_to_int:
                    # wint = dword_to_int[w]
                # else:
                    # wint = len(dword_to_int)
                    # dword_to_int[w] = wint
                    # int_to_dword[wint] = w
                # wintlist.append(wint)
            # word_to_dwordint[orig] = wintlist
        # if nondict:
            # slang_words = [wlist[i] for i in nondict]
            # dwords = [w for i, w in enumerate(wlist)
                      # if i not in nondict]
        # else:
            # slang_words = []
            # dwords = wlist
        # for w in slang_words:
            # slang_word_counts[w] += 1
        # for w in dwords:
            # word_counts[word] += 1

    # for f in filenames:
        # with open(f, 'r') as fo:
            # words = fo.read()\
                    # .replace("<eov>", " ")\
                    # .replace("\n", " ")\
                    # .replace('\xe2\x80\x99',"'")\
                    # .lower()\
                    # .translate(replace_punctuation)\
                    # .split()
            # for word in words:
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
                            # # TODO: check len of words is the same, and either difference is in
                            # # case of letter or different vowel
                            # if len(suggested) and lev.distance(w, suggested[0]) == 1:
                                # new_words.append(suggested[0])
                            # else:
                                # new_words.append(w)
                                # nondict.append(i)
                    # add_word(word, new_words, nondict=nondict)
    # end = time.time()
    # print "elapsed:", end - begin
    # return word_to_dwordint, dword_to_int, int_to_dword, word_counts, slang_word_counts


def word_count(filenames):
    counter = Counter([])
    replace_punctuation = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
    for f in filenames:
        with open(f, 'r') as fo:
            words = fo.read()\
                      .replace("<eov>", " ")\
                      .replace("\n", " ")\
                      .lower()\
                      .translate(replace_punctuation)\
                      .split()
            counter.update(words)
    return counter


def load_word_counts_into_db(collection, word_counts):
    swc = [{"word": w, "count": i} for w, i in word_counts.iteritems()]
    db[collection].insert_many(swc, ordered=False)
def load_word_to_int_dicts_into_db(word_to_dwordint, dword_to_int, int_to_dword):
    db.word_to_dwordint.insert_many([{"word": w, "int_list": wint}
                                     for w, wint in word_to_dwordint.iteritems()],
                                     ordered=False)
    db.dword_to_int.insert_many([{"word": w, "int": wint}
                                  for w, wint in dword_to_int.iteritems()],
                                  ordered=False)

if __name__ == '__main__':
    # manually get prons for top 300 or so
    # have flag for "is_acronym" that will jsut save whether it's an acronym
    # for the feature_extractor
    global client
    global db
    client = MongoClient()
    db = client['lil-neuron-db']
    #generate_files("data/lyric_files")
    # filenames = all_filenames("data/lyric_files")
    # result = build_word_dict(filenames)
    with open('word_stats_new2.p','r') as f:
        res = pck.load(f)
        # pck.dump({
            # 'word_to_dwordint': result[0],
            # 'dword_to_int': result[1],
            # 'int_to_dword': result[2],
            # 'word_counts': result[3],
            # 'slang_word_counts': result[4],
            # }, f)
    load_word_to_int_dicts_into_db(res['word_to_dwordint'],
                                   res['dword_to_int'],
                                   res['int_to_dword'])
    # load_word_counts_into_db("slang_words", res["slang_word_counts"])
    # load_word_counts_into_db("dwords", res["word_counts"])
    #filenames = all_filenames("data/lyric_files/Tyler, The Creator")
