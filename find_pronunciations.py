#from pymongo import MongoClient
#import pronouncing
import math
import multiprocessing as mp
import pdb
import string
import pronouncing
import cPickle as pck
all_phones = set(["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                  "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                  "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                  "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"])
def split_phone_stress(phone_stress):
    try:
        stress = int(phone_stress[-1])
    except:
        return phone_stress, 0
    else:
        return phone_stress[:-1], stress

def valid_pron(pron):
    for i, p in enumerate(pron):
        phone, stress = split_phone_stress(p)
        if stress > 2:
            print "Stress too high on {}th phone".format(i)
            return False
        elif phone not in all_phones:
            print "Unrecognized {}th phone: {}".format(i, phone)
            return False
    return True

def no_punc(w):
    if len(w.replace("'", "")) == 0:
        return False
    if w in ['nrp', 'eov', 'eos', 'eor']:
        return False
    for c in w:
        if c != "'" and c in string.punctuation:
            return False
    return True
    # [i for i in x if i in string.punctuation]
    # replace_punctuation = dict((ord(char), None) for char in string.punctuation
                               # if char != "'")
    # try:
        # new = unicode(w).translate(replace_punctuation)
    # except:
        # return False
    # else:
        # if len(new) == 0:
            # return False
    # return True

def find_pronunciations(from_word_dict_file=None):
    from pymongo import MongoClient
    client = MongoClient()
    db = client['lil-neuron-db']

    if from_word_dict_file:
        with open('data/snoop_word_dicts.p', 'rb') as f:
               dicts = pck.load(f)
               sw = dicts['slang_word_counts']
        SLANG_WORD_COUNTS = sorted([(k,v) for k,v in sw.iteritems() if v > 20 and no_punc(k)], key=lambda x: -x[1])
        records = db.slang_words.find({"pronunciation": {'$exists': False}})
        not_existing = set([r['word'] for r in records])
        SLANG_WORD_COUNTS = [(k,v) for k,v in SLANG_WORD_COUNTS if k in not_existing]
    else:
        records = db.slang_words.find({"pronunciation": {'$exists': False}})

        SLANG_WORD_COUNTS = sorted([(r["word"], r["count"]) for r in records
                                     if r["count"] > 60 and no_punc(r['word'])],
                                   key=lambda x: -x[1])

    print "SLANG WORDS LEFT:", len(SLANG_WORD_COUNTS)

    records = db.partial_words.find()
    PARTIAL_WORDS = {r['word']: r['pronunciation'] for r in records}

    session_count = 0
    for word, count in SLANG_WORD_COUNTS:
        inp = None
        pron = None
        while inp not in ['c', 'm']:
            print "SLANG WORD:"
            try:
                print "==> {} (count = {})".format(word, count)
            except UnicodeEncodeError:
                break
            print "Combine words (c) or manual (m):"

            inp = raw_input("==> ")
            if inp == 'c':
                print "Enter words:"
                words = raw_input("==> ").split()
                combined_prons = []
                for w in words:
                    while True:
                        existing_slang = db.slang_words.find({'word':w}).limit(1)
                        existing_slang = [s['pronunciation'] for s in existing_slang if 'pronunciation' in s]
                        if len(existing_slang):
                            prons = [' '.join(existing_slang[0])]
                        elif w in PARTIAL_WORDS:
                            prons = PARTIAL_WORDS[w]
                        else:
                            prons = pronouncing.phones_for_word(w)
                        print "{} prons:".format(w)
                        print prons
                        winp = raw_input("Select pron (int), new_word, start over (s), manual (m): ")
                        if not winp:
                            winp = 0
                        elif winp == 's':
                            # start over with new words
                            inp = None
                            break
                        elif winp == 'm':
                            while True:
                                pron = raw_input("Enter pron: ").split()
                                valid = valid_pron(pron)
                                if valid:
                                    break
                            string_pron = ' '.join(pron)
                            if w not in PARTIAL_WORDS:
                                PARTIAL_WORDS[w] = [string_pron]
                                db.partial_words.insert_one({'word': w, 'pronunciation': [string_pron]})
                            elif string_pron not in PARTIAL_WORDS[w]:
                                PARTIAL_WORDS[w].append(string_pron)
                                db.partial_words.update({'word': w}, {'$set': {'pronunciation': PARTIAL_WORDS[w]}})
                            combined_prons.extend(pron)
                            break
                        try:
                            pron_index = int(winp)
                        except:
                            w = winp
                        else:
                            wpron = prons[pron_index]
                            combined_prons.extend(wpron.split())
                            break
                    if not inp:
                        break
                pron = combined_prons
            elif inp == 'm':
                while True:
                    pron = raw_input("Enter pron: ").split()
                    valid = valid_pron(pron)
                    if valid:
                        break
            else:
                print "Unknown option"
            if pron:
                print pron
                print "updating"
                session_count += 1
                print "SESSION COUNT: {}".format(session_count)
                db.slang_words.update({'word': word}, {'$set':{'pronunciation':pron}})

def update_slang_ints():
    last_dwordint = sorted(INT_TO_DWORD.iterkeys(), reverse=True)[0]
    start = last_dwordint + 1
    records = db.slang_words.find()
    for i, r in enumerate(records):
        word = r['word']
        sym = i + start
        db.slang_words.update_one({'word': word}, {'$set': {'sym': sym}})
    # last_dwordint = sorted(INT_TO_DWORD.iterkeys(), reverse=True)[0]
    # start = last_dwordint + 1
    # for i, r in enumerate(db.slang_words.find()):
        # sym = i + start
        # db.slang_words.update({'word': r['word']}, {'sym': sym})

def _update_dword_prons(tuples):
    from pymongo import MongoClient
    import pronouncing
    client = MongoClient()
    db = client['lil-neuron-db']
    for sym, word, in tuples:
        prons = pronouncing.phones_for_word(word.lower())
        db.dword_to_int.update_one({'int': sym}, {'$set': {'prons': prons}})

def update_dword_prons(int_to_dword):
    groupsize = 1000
    ncores = mp.cpu_count()
    pool = mp.Pool(ncores)
    groups = []
    for i, t in enumerate(int_to_dword.iteritems()):
        if i % groupsize == 0:
            groups.append([])
        groups[-1].append(t)
    pool.map(_update_dword_prons, groups)

def load_dicts():
    # global client
    # global db
    # client = MongoClient()
    # db = client['lil-neuron-db']

    records = db.slang_words.find({"pronunciation": {'$exists': False}})
    global SLANG_WORD_COUNTS
    SLANG_WORD_COUNTS = sorted([(r["word"], r["count"]) for r in records
                                 if r["count"] > 60],
                               key=lambda x: -x[1])
    print "SLANG WORDS LEFT:", len(SLANG_WORD_COUNTS)
    # records = db.partial_words.find()
    # global PARTIAL_WORDS
    # # PARTIAL_WORDS = {r['word']: r['pronunciation'] for r in records}
    # PARTIAL_WORDS = {}

    records = db.dwords.find()
    global WORD_COUNTS
    WORD_COUNTS = sorted([(r["word"], r["count"]) for r in records],
                         key=lambda x: -x[1])


    # global WORD_TO_DWORD_INT
    # WORD_TO_DWORD_INT = {}
    # records = db.word_to_dwordint.find()
    # for r in records:
        # WORD_TO_DWORD_INT[r["word"]] = r["int_list"]

    global DWORD_TO_INT
    global INT_TO_DWORD
    DWORD_TO_INT = {}
    INT_TO_DWORD = {}
    records = db.dword_to_int.find()
    for r in records:
        INT_TO_DWORD[r["int"]] = r["word"]
if __name__ == '__main__':
    find_pronunciations()
    # load_dicts()
    #update_slang_ints()

    # global DWORD_TO_INT
    # global INT_TO_DWORD
    #DWORD_TO_INT = {}
    from pymongo import MongoClient
    client = MongoClient()
    db = client['lil-neuron-db']
    # INT_TO_DWORD = {}
    # records = db.dword_to_int.find()
    # for r in records:
        # INT_TO_DWORD[r["int"]] = r["word"]
    # update_dword_prons(INT_TO_DWORD)
